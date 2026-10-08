import gc
import os
import tkinter as tk
import unittest
from tkinter import ttk
from settings_dialog import open_settings


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Tk display is unavailable')
class DialogTests(unittest.TestCase):
    def test_read_only_project_values_remain_visible(self):
        root = tk.Tk()
        root.withdraw()
        try:
            window = open_settings(root, {'learning_rate': 0.000001}, trigger='subject03_person')
            window.withdraw()
            gc.collect()
            def descendants(parent):
                for child in parent.winfo_children():
                    yield child
                    yield from descendants(child)
            fields = [child for child in descendants(window) if isinstance(child, (ttk.Entry, ttk.Spinbox))]
            self.assertEqual(fields[0].get(), 'subject03_person')
            self.assertEqual(float(fields[1].get()), 0.000001)
            self.assertTrue(all(field.get() for field in fields))
        finally:
            root.destroy()
