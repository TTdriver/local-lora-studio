import tkinter as tk
from tkinter import ttk, messagebox
from training_settings import DEFAULTS, validate_settings
from core import validate_trigger


def open_settings(parent, values, on_save=None, title='New project training settings', trigger=None, note=None):
    window = tk.Toplevel(parent)
    window.title(title)
    window.geometry('760x780')
    window.minsize(680, 500)
    window.transient(parent)
    options = validate_settings(values)
    frame = ttk.Frame(window, padding=16)
    frame.pack(fill='both', expand=True)
    description = note or ('Settings are saved with the project before training starts.' if on_save else
                           'Saved project settings. Training has already started, so these are read-only.')
    ttk.Label(frame, text=description, wraplength=700).pack(anchor='w', pady=(0, 12))
    canvas = tk.Canvas(frame, highlightthickness=0)
    scroll = ttk.Scrollbar(frame, command=canvas.yview)
    scroll.pack(side='right', fill='y')
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(fill='both', expand=True)
    body = ttk.Frame(canvas, padding=(0, 0, 12, 0))
    item = canvas.create_window((0, 0), window=body, anchor='nw')
    body.bind('<Configure>', lambda _: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(item, width=event.width))
    body.columnconfigure(1, weight=1)
    variables, widgets = {}, []
    trigger_var = None
    offset = 0
    if trigger is not None:
        ttk.Label(body, text='Prompt trigger').grid(row=0, column=0, sticky='w')
        trigger_var = tk.StringVar(value=trigger)
        entry = ttk.Entry(body, textvariable=trigger_var)
        entry.grid(row=0, column=1, sticky='ew')
        widgets.append(entry)
        ttk.Label(body, text='Use this exact phrase in your captions and image prompts. Blank uses the project alias + _person.', wraplength=650).grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 12))
        offset = 2
    fields = [
        ('learning_rate', 'Learning rate', 'How quickly the model learns. Default: 0.0001.', 0.000001, 0.001, 0.00001),
        ('rank', 'LoRA rank', 'Capacity to learn detail. Higher values require more memory.', 8, 128, 8),
        ('alpha', 'LoRA alpha', 'Scales LoRA updates. Default: 32, matching the default rank.', 1, 128, 1),
        ('batch_size', 'Batch size', 'Photos per update. Values above 1 require more GPU memory.', 1, 4, 1),
        ('caption_dropout', 'Caption dropout (%)', 'Percentage of training examples that omit captions.', 0, 100, 1),
        ('checkpoint_interval', 'Checkpoint / preview interval', 'Save a checkpoint and generate previews every this many steps.', 25, 4000, 25),
        ('keep_checkpoints', 'Keep intermediate checkpoints', 'Number of intermediate checkpoints retained; the final file is also saved.', 1, 20, 1),
        ('preview_seed', 'Preview seed', 'Fixed seed for comparing test pictures between checkpoints.', 0, 4294967295, 1),
    ]
    for index, (key, label, help_text, low, high, increment) in enumerate(fields):
        row = index + offset // 2
        ttk.Label(body, text=label).grid(row=row * 2, column=0, sticky='w', padx=(0, 16))
        value = options[key] * 100 if key == 'caption_dropout' else options[key]
        var = tk.StringVar(value=str(value))
        variables[key] = var
        widget = ttk.Spinbox(body, textvariable=var, from_=low, to=high, increment=increment, width=18)
        widget.grid(row=row * 2, column=1, sticky='ew')
        widgets.append(widget)
        ttk.Label(body, text=help_text, wraplength=650).grid(row=row * 2 + 1, column=0, columnspan=2, sticky='w', pady=(2, 12))
    row = len(fields) * 2 + offset
    ttk.Label(body, text='Training resolutions').grid(row=row, column=0, sticky='w')
    sizes = ttk.Frame(body)
    sizes.grid(row=row, column=1, sticky='w')
    resolutions = {}
    for size in (512, 768, 1024):
        var = tk.BooleanVar(value=size in options['resolutions'])
        resolutions[size] = var
        widget = ttk.Checkbutton(sizes, text=str(size), variable=var)
        widget.pack(side='left', padx=(0, 12))
        widgets.append(widget)
    ttk.Label(body, text='Choose one or more sizes. Larger resolutions require more memory.', wraplength=650).grid(row=row+1, column=0, columnspan=2, sticky='w', pady=(2, 12))
    ttk.Label(body, text='Preview prompts — one per line; use {trigger} for your project’s trigger.', wraplength=650).grid(row=row+2, column=0, columnspan=2, sticky='w')
    prompts = tk.Text(body, height=5, wrap='word', font=('Sans', 10))
    prompts.insert('1.0', '\n'.join(options['preview_prompts']))
    prompts.grid(row=row+3, column=0, columnspan=2, sticky='ew', pady=8)
    ttk.Label(body, text='Defaults match the working preset. Higher batch size, rank or resolution can exceed available GPU memory.', wraplength=650).grid(row=row+4, column=0, columnspan=2, sticky='w', pady=(0, 12))
    buttons = ttk.Frame(window, padding=12)
    buttons.pack(fill='x')
    ttk.Button(buttons, text='Close', command=window.destroy).pack(side='right')

    def restore_defaults():
        for key, var in variables.items():
            var.set(str(DEFAULTS[key] * 100 if key == 'caption_dropout' else DEFAULTS[key]))
        for size, var in resolutions.items():
            var.set(size in DEFAULTS['resolutions'])
        prompts.delete('1.0', 'end')
        prompts.insert('1.0', '\n'.join(DEFAULTS['preview_prompts']))

    def apply():
        try:
            values = {key: var.get() for key, var in variables.items()}
            values['caption_dropout'] = float(values['caption_dropout']) / 100
            values['resolutions'] = [size for size, var in resolutions.items() if var.get()]
            values['preview_prompts'] = [line.strip() for line in prompts.get('1.0', 'end').splitlines() if line.strip()]
            settings = validate_settings(values)
            if trigger_var is not None:
                on_save(settings, validate_trigger(trigger_var.get().strip()))
            else:
                on_save(settings)
        except (ValueError, OSError) as error:
            messagebox.showerror('Training settings', str(error), parent=window)
            return
        window.destroy()

    if on_save:
        ttk.Button(buttons, text='Save settings', command=apply).pack(side='right', padx=8)
        ttk.Button(buttons, text='Reset to defaults', command=restore_defaults).pack(side='left')
    else:
        for widget in widgets:
            widget.configure(state='disabled')
        prompts.configure(state='disabled')
    # Wheel bindings stay within this dialog; the prompt editor keeps its own scrolling.
    def wheel(event):
        if event.widget is not prompts:
            amount = (-1 if event.num == 4 else 1) if event.num in (4, 5) else -int(event.delta / 120)
            canvas.yview_scroll(amount, 'units')
    for sequence in ('<Button-4>', '<Button-5>', '<MouseWheel>'):
        window.bind(sequence, wheel)
    # Tk widgets keep Tcl variable names, not Python objects. Read-only dialogs
    # have no Save callback retaining these variables, so hold them for the window.
    window.setting_variables = (variables, resolutions, trigger_var)
    return window
