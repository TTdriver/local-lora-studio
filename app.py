#!/usr/bin/python3
"""Local LoRA Studio: import photos, train privately, and activate a checkpoint."""
import json
import base64
import io
import os
import queue
import re
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image
import psutil
from progress import describe, log_tail
from core import STATE, IMAGE, ensure_state, prepare, resource_errors, runtime_ready, copy_project, edit_project_settings
from training_settings import validate_settings
from settings_dialog import open_settings

HERE = Path(__file__).resolve().parent
from tk_update_link import UpdateLink

APP_VERSION = '0.1.1'
UPDATE_VERSION_URL = 'https://api.github.com/repos/TTdriver/local-lora-studio/contents/VERSION'
DOWNLOAD_URL = 'https://github.com/TTdriver/local-lora-studio#installation'

BG, CARD, FG, MUTED, ACCENT = '#14171e', '#242b36', '#edf2fa', '#a1adbf', '#76d8ae'


class App:
    def __init__(self):
        ensure_state()
        self.root = tk.Tk(className='LocalLoraStudio')
        self.root.title('Local LoRA Studio')
        self.app_icon = tk.PhotoImage(file=str(Path(__file__).resolve().parent / 'app-icon.png'))
        self.root.iconphoto(True, self.app_icon)
        self.root.geometry('1060x940')
        self.root.minsize(1000, 900)
        self.root.configure(bg=BG)
        self.paths, self.jobs, self.selected, self.preview_image = [], [], None, None
        self.events, self.busy = queue.Queue(), False
        self.new_training_settings = validate_settings()
        self.new_trigger = ''
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TProgressbar', troughcolor=CARD, background=ACCENT)
        self.update_notice = UpdateLink(self.root, APP_VERSION, UPDATE_VERSION_URL, DOWNLOAD_URL, 'LocalLoRAStudio', BG, MUTED)
        outer = tk.Frame(self.root, bg=BG, padx=24, pady=20)
        outer.pack(fill='both', expand=True)
        self.label(outer, 'LOCAL LoRA STUDIO', 20, FG).pack(anchor='w')
        self.label(outer, 'Add photos → automatic captions → training → personal LoRA', 11).pack(anchor='w', pady=(4, 12))
        self.resources = self.label(outer, '', 10)
        self.resources.pack(anchor='w', pady=(0, 12))
        body = tk.Frame(outer, bg=BG)
        body.pack(fill='both', expand=True)
        left = tk.Frame(body, bg=BG, width=310)
        left.pack(side='left', fill='y', padx=(0, 20))
        self.label(left, '1. Add your photos', 13, FG).pack(anchor='w')
        self.button(left, 'Add photos…', self.add_photos).pack(fill='x', pady=(8, 4))
        self.button(left, 'Clear selected photos', self.clear_photos).pack(fill='x')
        self.photo_count = self.label(left, 'No photos selected. Aim for 20–30.', 10)
        self.photo_count.pack(anchor='w', pady=8)
        self.label(left, 'Project label (use an alias)', 10).pack(anchor='w')
        self.name = self.entry(left, 'subject01')
        self.label(left, 'Training length', 10).pack(anchor='w', pady=(10, 0))
        self.steps = tk.StringVar(value='2000')
        ttk.Combobox(left, textvariable=self.steps, values=['1000', '1500', '2000', '3000'],
                     state='readonly', width=28).pack(fill='x', pady=4)
        self.auto_install = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text='Activate finished LoRA in Image Studio', variable=self.auto_install,
                       bg=BG, fg=FG, selectcolor=CARD, activebackground=BG,
                       activeforeground=FG).pack(anchor='w', pady=8)
        self.button(left, 'New project settings…', self.new_settings).pack(fill='x', pady=4)
        self.button(left, 'Create project', self.create).pack(fill='x', pady=4)
        self.label(left, '2. Train your likeness', 13, FG).pack(anchor='w', pady=(18, 5))
        self.train_button = self.button(left, 'Start automatic training', self.start)
        self.train_button.pack(fill='x', pady=4)
        self.button(left, 'Cancel selected job', self.cancel).pack(fill='x', pady=4)
        self.button(left, 'Prepare / repair trainer', self.setup).pack(fill='x', pady=4)
        self.label(left, 'Training needs the GPU to itself.\nChat and images pause during a job.\nBoth services restart afterward.\n\nPhotos and captions stay on this VM.\nFirst use downloads model weights.\nClosing this window keeps training running.', 10).pack(anchor='w', pady=12)
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)
        self.label(right, 'Projects', 13, FG).pack(anchor='w')
        self.listbox = tk.Listbox(right, bg=CARD, fg=FG, selectbackground='#3c5266', height=4,
                                 font=('Sans', 11), highlightthickness=0, exportselection=False)
        self.listbox.pack(fill='x', pady=6)
        self.listbox.bind('<<ListboxSelect>>', self.select)
        self.status = self.label(right, 'Choose a project or add photos to begin.', 11, FG)
        self.status.config(wraplength=535, justify='left')
        self.status.pack(anchor='w', pady=(6, 10))
        self.progress = ttk.Progressbar(right, maximum=100)
        self.progress.pack(fill='x')
        self.details = self.label(right, '', 10, FG)
        self.details.config(wraplength=610)
        self.details.pack(anchor='w', pady=(8, 6))
        self.label(right, 'Generated training preview', 12, FG).pack(anchor='w')
        self.preview_caption = self.label(right, 'Test pictures show how the LoRA changes at saved training steps.', 10)
        self.preview_caption.config(wraplength=610)
        self.preview_caption.pack(anchor='w', pady=4)
        self.preview = tk.Label(right, bg=CARD, fg=MUTED, text='Training previews will appear here', height=8)
        self.preview.pack(fill='both', expand=True, pady=12)
        self.preview_path = None
        self.checkpoints = ttk.Combobox(right, state='readonly')
        self.checkpoints.pack(fill='x')
        self.checkpoint_paths = []
        actions = tk.Frame(right, bg=BG)
        actions.pack(fill='x', pady=8)
        self.button(actions, 'Activate checkpoint', self.activate).pack(side='left')
        self.button(actions, 'View live log', self.open_log).pack(side='left', padx=8)
        folders = tk.Frame(right, bg=BG)
        folders.pack(fill='x')
        self.button(folders, 'Open project folder', self.open_job).pack(side='left')
        self.button(folders, 'Project settings…', self.project_settings).pack(side='left', padx=8)
        self.button(right, 'New run with saved captions', self.copy_run).pack(anchor='w', pady=4)
        self.button(actions, 'Restore AI services', self.restore).pack(side='left')
        self.note = self.label(outer, '', 10)
        self.note.pack(anchor='w', pady=(10, 0))
        self.root.after(100, self.refresh)

    def label(self, parent, text, size=10, color=MUTED):
        return tk.Label(parent, text=text, bg=BG, fg=color, font=('Sans', size), anchor='w', justify='left')

    def button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=CARD, fg=FG,
                         activebackground='#354354', activeforeground=FG, relief='flat',
                         padx=12, pady=8, font=('Sans', 10))

    def entry(self, parent, default):
        widget = tk.Entry(parent, bg=CARD, fg=FG, insertbackground=FG, relief='flat', font=('Sans', 11))
        widget.insert(0, default)
        widget.pack(fill='x', ipady=6, pady=4)
        return widget

    def add_photos(self):
        paths = filedialog.askopenfilenames(title='Choose photos of the same person',
            filetypes=[('Photos', '*.jpg *.jpeg *.png *.webp *.bmp'), ('All files', '*')])
        self.paths = list(dict.fromkeys([*self.paths, *paths]))
        self.photo_count.config(text=f'{len(self.paths)} photos selected. Originals stay unchanged.')

    def clear_photos(self):
        self.paths = []
        self.photo_count.config(text='No photos selected. Aim for 20–30.')

    def background(self, operation):
        if self.busy:
            messagebox.showinfo('Working', 'Wait for the current operation to finish.')
            return
        self.busy = True
        def run():
            try:
                result = operation()
                self.events.put(('result', result))
            except Exception as error:
                self.events.put(('error', str(error)))
        threading.Thread(target=run, daemon=True).start()

    def create(self):
        if not self.paths:
            messagebox.showinfo('Photos', 'Add photos first.')
            return
        paths, name, steps, install = list(self.paths), self.name.get().strip(), int(self.steps.get()), self.auto_install.get()
        self.note.config(text='Preparing photos and removing duplicate images…')
        settings = validate_settings(self.new_training_settings)
        trigger = self.new_trigger
        self.background(lambda: ('project', prepare(paths, name, steps, install, settings, trigger)))

    def new_settings(self):
        def apply(settings, trigger):
            self.new_training_settings = settings
            self.new_trigger = trigger
            self.note.config(text='Training settings saved for the next project you create.')
        open_settings(self.root, self.new_training_settings, apply, trigger=self.new_trigger)

    def copy_run(self):
        if not self.selected:
            messagebox.showinfo('New run', 'Select the project with your edited captions first.')
            return
        job = self.selected
        if self.job_running(job):
            messagebox.showinfo('New run', 'Wait for this training run to finish before copying it.')
            return
        self.background(lambda: ('project', copy_project(job)))

    def project_settings(self):
        if not self.selected:
            messagebox.showinfo('Project settings', 'Select or create a project first.')
            return
        job = self.selected
        info = json.loads((job / 'job.json').read_text())
        data = json.loads((job / 'status.json').read_text())
        stopped = data['stage'] in ('cancelled', 'failed') and not data.get('pid')
        editable = ((data['stage'] == 'ready' and not data.get('started')) or stopped) and not self.job_running(job)
        def apply(settings, trigger):
            if self.busy or self.job_running(job):
                raise ValueError('Wait for the current operation to finish before saving.')
            self.background(lambda: ('settings_project', edit_project_settings(job, settings, trigger)))
        open_settings(self.root, info.get('training_settings'), apply if editable else None,
                      title='Project settings — ' + info['name'], trigger=info['trigger'],
                      note='Saving creates a fresh run with your photos and captions, excluding old checkpoints and caches. Click Start automatic training afterward.' if stopped else None)

    def start(self):
        if not self.selected:
            if self.paths:
                issues = resource_errors()
                if issues:
                    messagebox.showinfo('Increase VM resources', '\n\n'.join(issues))
                    return
                paths, name, steps, install = list(self.paths), self.name.get().strip(), int(self.steps.get()), self.auto_install.get()
                settings = validate_settings(self.new_training_settings)
                trigger = self.new_trigger
                self.background(lambda: ('start_project', prepare(paths, name, steps, install, settings, trigger)))
            else:
                messagebox.showinfo('Photos', 'Add photos or select an existing project first.')
            return
        issues = resource_errors()
        if issues:
            messagebox.showinfo('Increase VM resources', '\n\n'.join(issues))
            return
        if not runtime_ready():
            self.setup()
            return
        if self.job_running(self.selected):
            messagebox.showinfo('Training', 'This job is already running.')
            return
        if json.loads((self.selected / 'status.json').read_text())['stage'] == 'complete':
            messagebox.showinfo('Fresh training run', 'Use New run with saved captions to train again. It keeps your edits and starts without the old checkpoints or caption caches.')
            return
        if any(self.job_running(job) for job in self.jobs):
            messagebox.showinfo('Training', 'One training job can run at a time.')
            return
        with (self.selected / 'worker.log').open('a') as log:
            subprocess.Popen(['/usr/bin/python3', str(HERE / 'worker.py'), str(self.selected)],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.note.config(text='Starting background job…')

    @staticmethod
    def job_running(job):
        try:
            data = json.loads((job / 'status.json').read_text())
            pid = data.get('pid')
            return bool(pid and str(job).encode() in Path(f'/proc/{pid}/cmdline').read_bytes())
        except (OSError, ValueError):
            return False

    def cancel(self):
        if self.selected and self.job_running(self.selected):
            data = json.loads((self.selected / 'status.json').read_text())
            os.kill(data['pid'], signal.SIGTERM)
            self.note.config(text='Cancelling; waiting for the checkpoint write and service restoration…')

    def setup(self):
        issues = resource_errors(training=False)
        if issues:
            messagebox.showinfo('Disk space', '\n'.join(issues)); return
        def build():
            log_path = STATE / 'runtime-setup.log'
            with log_path.open('w') as log:
                result = subprocess.run(['docker', 'build', '-t', IMAGE, str(HERE)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Trainer setup failed. Log: {log_path}')
            return ('message', 'Trainer runtime prepared.')
        self.note.config(text='Preparing trainer. This can take several minutes…')
        self.background(build)

    def select(self, _=None):
        indices = self.listbox.curselection()
        if indices:
            self.selected = self.jobs[indices[0]]
            self.preview_path = None
            self.preview.config(image='', text='No generated preview yet', height=8)
            self.preview_caption.config(text='Test pictures show how the LoRA changes at saved training steps.')

    def activate(self):
        index = self.checkpoints.current()
        if self.selected and index >= 0:
            job, checkpoint = self.selected, self.checkpoint_paths[index]
            def operation():
                result = subprocess.run(['/usr/bin/python3', str(HERE / 'worker.py'), str(job), '--install', str(checkpoint)],
                                         capture_output=True, text=True, timeout=90)
                if result.returncode:
                    raise RuntimeError(result.stderr[-1500:])
                return ('message', 'Checkpoint activated. Include the project trigger in image prompts.')
            self.background(operation)

    def open_job(self):
        if self.selected:
            subprocess.Popen(['xdg-open', str(self.selected)])

    def open_log(self):
        if not self.selected:
            return
        job = self.selected
        window = tk.Toplevel(self.root)
        window.title('Training log — ' + job.name)
        window.geometry('900x560')
        text = tk.Text(window, wrap='word', bg=CARD, fg=FG, font=('Monospace', 10))
        scroll = ttk.Scrollbar(window, command=text.yview)
        scroll.pack(side='right', fill='y')
        text.config(yscrollcommand=scroll.set)
        text.pack(fill='both', expand=True)
        previous = None
        def update():
            nonlocal previous
            if not window.winfo_exists():
                return
            content = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', log_tail(job))
            if content != previous:
                at_end = text.yview()[1] >= 0.98
                position = text.yview()[0]
                text.config(state='normal')
                text.delete('1.0', 'end')
                text.insert('1.0', content or 'No training log yet.')
                text.config(state='disabled')
                text.yview_moveto(1 if at_end else position)
                previous = content
            window.after(1500, update)
        update()

    def restore(self):
        if any(self.job_running(job) for job in self.jobs):
            messagebox.showinfo('Training', 'Wait for or cancel training before restoring image generation.'); return
        def restore_services():
            names = ['ollama-pocket-comfyui']
            for job in self.jobs:
                data = json.loads((job / 'status.json').read_text())
                if data.get('paused_ollama'):
                    names.append('ollama')
                    break
            result = subprocess.run(['docker', 'start', *names], capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError('Could not restore all AI services. Check Docker and retry.')
            from core import status
            for job in self.jobs:
                data = json.loads((job / 'status.json').read_text())
                status(job, data['stage'], data['message'], paused_comfy=False, paused_ollama=False)
            return ('message', 'AI services restarted.')
        self.background(restore_services)

    def refresh(self):
        try:
            while True:
                kind, result = self.events.get_nowait()
                self.busy = False
                if kind == 'error':
                    self.note.config(text='Operation failed.'); messagebox.showerror('Local LoRA Studio', result)
                elif result[0] in ('project', 'start_project', 'settings_project'):
                    self.selected = result[1]
                    self.note.config(text='Project ready. Existing captions are kept; only missing captions are generated.')
                    if result[0] == 'settings_project':
                        self.note.config(text='Settings saved. Project ready to start; saved photos and captions preserved.')
                    if result[0] == 'start_project':
                        self.start()
                else:
                    self.note.config(text=result[1])
        except queue.Empty:
            pass
        jobs = sorted((STATE / 'jobs').glob('*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        paths = [p.parent for p in jobs]
        if paths != self.jobs:
            self.jobs = paths
            self.listbox.delete(0, 'end')
            for path in paths:
                info = json.loads((path / 'job.json').read_text())
                self.listbox.insert('end', f'{info["name"]} · {info["photos"]} photos')
        if self.selected is None and self.jobs:
            self.selected = self.jobs[0]
        if self.selected in self.jobs:
            self.listbox.selection_clear(0, 'end')
            self.listbox.selection_set(self.jobs.index(self.selected))
            try:
                data = json.loads((self.selected / 'status.json').read_text())
                info = json.loads((self.selected / 'job.json').read_text())
                message = data['message']
                if data.get('pid') and not self.job_running(self.selected) and data['stage'] not in ('complete', 'failed', 'cancelled'):
                    message = 'Job was interrupted. Saved files remain; restore image service before retrying.'
                self.status.config(text=f'{data["stage"].upper()} · Trigger: {info["trigger"]}\n{message}')
                self.details.config(text=describe(self.selected, info, data))
                self.progress['value'] = 100 * data.get('step', 0) / info['steps']
                checkpoints = sorted((self.selected / 'output').rglob('*.safetensors'))
                if checkpoints != self.checkpoint_paths:
                    self.checkpoint_paths = checkpoints
                    self.checkpoints['values'] = [p.name for p in checkpoints]
                    if checkpoints:
                        self.checkpoints.current(len(checkpoints) - 1)
                samples = [p for p in list((self.selected / 'output').rglob('*.png')) + list((self.selected / 'output').rglob('*.jpg')) if '.thumbs' not in p.parts]
                if samples:
                    newest = max(samples, key=lambda p: p.stat().st_mtime)
                    if newest != self.preview_path:
                        with Image.open(newest) as image:
                            image.thumbnail((440, 240))
                            buffer = io.BytesIO()
                            image.save(buffer, format='PNG')
                            self.preview_image = tk.PhotoImage(data=base64.b64encode(buffer.getvalue()))
                        self.preview.config(image=self.preview_image, text='', height=0)
                        self.preview_path = newest
                        match = re.search(r'__(\d+)_(\d+)', newest.stem)
                        step, index = (int(match[1]), int(match[2])) if match else (None, None)
                        config = json.loads((self.selected / 'config.json').read_text())
                        prompts = config['config']['process'][0]['sample']['prompts']
                        prompt = prompts[index] if index is not None and index < len(prompts) else ''
                        self.preview_caption.config(text=f'Generated at step {step:,} · {prompt}' if step is not None else newest.name)
            except (OSError, ValueError, KeyError):
                pass
        ram = psutil.virtual_memory()
        memory = ram.total / 1024**3
        available = ram.available / 1024**3
        import shutil
        free = shutil.disk_usage(STATE).free / 1024**3
        self.resources.config(text=f'VM RAM: {available:.1f} / {memory:.1f} GB available · Disk free: {free:.1f} GB · GPU: RTX 3090 · Private local training')
        self.root.after(1000, self.refresh)


if __name__ == '__main__':
    App().root.mainloop()
