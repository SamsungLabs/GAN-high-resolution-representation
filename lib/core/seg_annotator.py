import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import cv2
import numpy as np
import time
from os import makedirs
from os.path import join, isdir
import random
import pickle
from lib.core.seg_solver import SegSolver
from lib.utils.utils import get_draw_mask
from lib.core.image_generator import ImageGenerator
from tqdm import tqdm


class SegmentationAnnotator(tk.Frame):
    def __init__(self, parent, root_dir, gan_gpu_ids, solver_gpu_ids, gan_dir, gan='ffhq'):
        tk.Frame.__init__(self, parent)
        self.master.title('Image Viewer')

        self.root_dir = root_dir
        self.initialize_from_root()

        fram = tk.Frame(self)
        fram.pack(side=tk.BOTTOM, fill=tk.BOTH)
        self.ok_btn = tk.Button(fram, text='OK', command=self.on_ok_clicked)
        self.skip_btn = tk.Button(fram, text='Skip', command=self.on_skip_clicked)
        self.retrain_btn = tk.Button(fram, text='Retrain', command=self.on_train_clicked)
        self.generate_btn = tk.Button(fram, text='Generate', command=self.on_generate_clicked)
        self.pseudo_btn = tk.Button(fram, text='Pseudo-Labeling', command=self.on_psuedo_labeling_clicked)

        self.ok_btn.pack(side=tk.RIGHT)
        self.skip_btn.pack(side=tk.RIGHT)
        self.retrain_btn.pack(side=tk.RIGHT)
        self.generate_btn.pack(side=tk.RIGHT)
        self.pseudo_btn.pack(side=tk.RIGHT)

        self.fram = fram

        self.la = tk.Label(self)
        self.la.pack()

        self.can = tk.Canvas(self, cursor='none')
        self.can.bind('<Motion>', self.on_mouse_move)
        self.can.bind('<ButtonPress-1>', self.on_mouse_down)
        self.can.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.can.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.can.bind('<Button-4>', self.on_mouse_wheel)
        self.can.bind('<Button-5>', self.on_mouse_wheel)
        self.can.bind('<Leave>', self.on_mouse_leave)
        self.can.pack()

        parent.bind('<KeyPress>', self.on_key_down)
        parent.bind('<KeyRelease>', self.on_key_up)

        self.mouse_down = False
        self.prev_positiov = None
        self.width = 20.
        self.ctrl = False
        self.alt = False
        self.shift = False
        self.buffer_img = None
        self.draw = None
        self.has_changes = False
        self.history = []
        self.cursor = None
        self.prev_cursor_pos = None, None
        self.mouse_down_history_id = None
        self.mouse_up_history_id = None

        self.netG = ImageGenerator(gpu_ids=gan_gpu_ids, gan_dir=gan_dir, gan=gan)
        self.solver = SegSolver(self.netG.max_res_log2, join(self.root_dir, 'data'),
                                join(self.root_dir, 'checkpoints'), gpu_ids=solver_gpu_ids)
        self.image_iterator = self.create_image_iterator()

        if self.solver.is_trained:
            self.generate_btn.config(state='normal')
            self.pseudo_btn.config(state='normal')
        else:
            self.generate_btn.config(state='disabled')
            self.pseudo_btn.config(state='disabled')

        self.next_image()

    def remove_last_elements_from_history(self, num_elements=1):

        num_elements = min(len(self.history), num_elements)

        if num_elements < 1:
            return

        last_elements = self.history[-num_elements:]
        for item in last_elements[::-1]:
            item1, item2, item3 = item

            if item1 is not None:
                self.can.delete(item1[0])
            if item2 is not None:
                self.can.delete(item2[0])
            if item3 is not None:
                self.can.delete(item3[0])

        self.history = self.history[:-num_elements]

    def prepare_drawed_mask(self):

        self.draw = ImageDraw.Draw(self.buffer_img)
        for item1, item2, item3 in self.history:
            if item1 is not None:
                x0, y0, x1, y1, width, color1 = item1[1]
                self.draw.line([x0, y0, x1, y1], color1, width=width)

            if item2 is not None:
                xs0, ys0, xs1, ys1, color2 = item2[1]
                self.draw.ellipse([xs0, ys0, xs1, ys1], fill=color2, outline=None)

            if item3 is not None:
                xe0, ye0, xe1, ye1, color3 = item3[1]
                self.draw.ellipse([xe0, ye0, xe1, ye1], fill=color3, outline=None)

    def on_key_down(self, event):
        k = event.keycode

        self.ctrl = self.ctrl or k == 37
        self.alt = self.alt or k == 50
        self.shift = self.shift or k == 64
        z_pressed_now = k == 52

        if self.ctrl:
            self.update_cursor()

        if z_pressed_now and self.ctrl:
            if self.mouse_up_history_id is not None and self.mouse_down_history_id is not None:
                last_action_len = self.mouse_up_history_id - self.mouse_down_history_id
                if last_action_len > 0:
                    self.remove_last_elements_from_history(num_elements=last_action_len)

    def on_key_up(self, event):
        k = event.keycode

        ctrl = k == 37
        alt = k == 50
        shift = k == 64

        prev_ctrl = self.ctrl

        if ctrl:
            self.ctrl = False
        if alt:
            self.alt = False
        if shift:
            self.shift = False

        if prev_ctrl != self.ctrl:
            self.update_cursor()

    def on_mouse_leave(self, event):
        self.update_cursor(event, disable=True)

    def on_mouse_wheel(self, event):
        if event.num == 4:
            coeff = 1.2
        else:
            coeff = 1 / 1.2

        self.width = self.width * coeff
        self.width = max(1., min(200., self.width))
        self.update_cursor()

    def update_cursor(self, event=None, disable=False):
        if self.cursor is not None:
            self.can.delete(self.cursor)

        if not disable:
            color_display = '#f0f0f0' if not self.ctrl else '#8f8f8f'
            if event is not None:
                x, y = event.x, event.y
            else:
                x, y = self.prev_cursor_pos
            if x is None or y is None:
                return
            xs0, ys0 = x - int(self.width / 2), y - int(self.width / 2)
            xs1, ys1 = x + int(self.width / 2), y + int(self.width / 2)
            self.cursor = self.can.create_oval(xs0, ys0, xs1, ys1, outline=color_display, width=3)
            self.prev_cursor_pos = x, y

    def draw_event(self, pos):
        color_display = '#ffffff' if not self.ctrl else '#808080'
        color = '#ffffff' if not self.ctrl else '#808080'

        if self.prev_positiov is not None:
            x0, y0 = self.prev_positiov
            x1, y1 = pos

            id = self.can.create_line(x0, y0, x1, y1, width=int(self.width), fill=color_display)
            item1 = [id, (x0, y0, x1, y1, int(self.width), color)]

            xs0, ys0 = x0 - int(self.width / 2), y0 - int(self.width / 2)
            xs1, ys1 = x0 + int(self.width / 2), y0 + int(self.width / 2)
            id = self.can.create_oval(xs0, ys0, xs1, ys1, fill=color_display, width=0)
            item2 = [id, (xs0, ys0, xs1, ys1, color)]

            xe0, ye0 = x1 - int(self.width / 2), y1 - int(self.width / 2)
            xe1, ye1 = x1 + int(self.width / 2), y1 + int(self.width / 2)
            id = self.can.create_oval(xe0, ye0, xe1, ye1, fill=color_display, width=0)
            item3 = [id, (xe0, ye0, xe1, ye1, color)]

            self.history.append([item1, item2, item3])
            self.has_changes = True

        else:

            x0, y0 = pos
            item1 = None

            xs0, ys0 = x0 - int(self.width / 2), y0 - int(self.width / 2)
            xs1, ys1 = x0 + int(self.width / 2), y0 + int(self.width / 2)
            id = self.can.create_oval(xs0, ys0, xs1, ys1, fill=color_display, width=0)
            item2 = [id, (xs0, ys0, xs1, ys1, color)]

            item3 = None

            self.history.append([item1, item2, item3])
            self.has_changes = True

        self.prev_positiov = (pos)

    def on_mouse_move(self, event):
        self.update_cursor(event)

        if self.mouse_down:
            pos = (event.x, event.y)
            self.draw_event(pos)

    def on_mouse_down(self, event):
        self.mouse_down = True
        self.mouse_down_history_id = len(self.history)
        pos = (event.x, event.y)
        self.draw_event(pos)

    def on_mouse_up(self, event):
        self.mouse_down = False
        self.mouse_up_history_id = len(self.history)
        self.prev_positiov = None

    def on_train_clicked(self):
        if self.has_changes:
            self.save_current_results()
        self.toggle_disable_main()
        time.sleep(1)


        def epoch_end_callback():
            mask, _ = self.solver.predict(self.features)
            mask = mask[0].astype(np.uint8)
            # mask[:, :, 0] = morph_mask(mask[:, :, 0])
            img = get_draw_mask(self.img_orig, mask[:, :, 0], alpha=0.5, color_map=None, skip_background=True)
            self.img_frame = ImageTk.PhotoImage(Image.fromarray(img))
            self.can.config(bg='#000000', width=self.img_frame.width(), height=self.img_frame.height())
            self.can.create_image(0, 0, image=self.img_frame, anchor=tk.NW)
            self.can.update()

        self.solver.fit(epoch_end_callback)
        self.on_train_finished()

        # pool = Pool(processes=1)  # Start a worker processes.
        # pool.apply_async(long_compute, args=(2,), callback=callback)  # Evaluate "f(10)" asynchronously calling callback when finished.
        # pool.close()

        # self.win = tk.Toplevel(width=300, height=300)
        # self.win.wm_title('Training options')

        # l = tk.Label(self.win, text='Input')
        # b = tk.Button(self.win, text='Okay', command=self.on_train_finished)

        # l.pack(side=tk.RIGHT)
        # b.pack(side=tk.RIGHT)

    def on_train_finished(self):
        print('train finished.')
        self.toggle_disable_main(True)
        self.next_image()

    def toggle_disable_main(self, enabled=False):
        state = 'normal' if enabled else 'disabled'
        self.ok_btn.config(state=state)
        self.skip_btn.config(state=state)
        self.retrain_btn.config(state=state)
        if self.solver.is_trained:
            self.generate_btn.config(state=state)
            self.pseudo_btn.config(state=state)
        else:
            self.generate_btn.config(state='disabled')
            self.pseudo_btn.config(state='disabled')

    # def on_train_finished(self):
    #     print('train finished.')
    #     self.toggle_disable_main(enabled=True)
    #     self.win.destroy()

    def on_skip_clicked(self):
        self.next_image()

    def on_ok_clicked(self):
        if self.has_changes:
            self.save_current_results()
        self.next_image()

    def on_generate_clicked(self):

        self.toggle_disable_main(enabled=False)
        time.sleep(1)

        n_imgs = 10000
        dst_dir = join(self.root_dir, 'generated')
        with tqdm(total=n_imgs) as pb:
            for i in range(n_imgs):
                img, mask, features, _ = next(self.image_iterator)
                imname = f'img_{i:06d}.jpg'
                visname = f'vis_{i:06d}.jpg'
                maskname = f'mask_{i:06d}.png'
                img_vis = get_draw_mask(img, mask[:, :, 0], alpha=0.5, color_map=None, skip_background=True)
                # cv2.imwrite(join(dst_dir, imname), img[:,:,::-1])
                cv2.imwrite(join(dst_dir, visname), img_vis[:,:,::-1])
                # cv2.imwrite(join(dst_dir, maskname), mask[:,:,0])
                pb.update()

        self.toggle_disable_main(enabled=True)

    def on_psuedo_labeling_clicked(self):

        self.toggle_disable_main(enabled=False)
        time.sleep(1)

        n_imgs = 10
        dst_dir = join(self.root_dir, 'data')
        with tqdm(total=n_imgs) as pb:
            for i in range(n_imgs):
                img, mask, features, probs = next(self.image_iterator)
                imname = f'img_{i:06d}.jpg'
                maskname = f'mask_{i:06d}.png'
                feature_name = f'feat_{i:06d}.pickle'

                mask = np.zeros((probs.shape[1], probs.shape[2], 1), dtype=np.uint8)
                mask[probs[1] > 0.85] = 255
                mask[probs[1] < 0.01] = 128

                cv2.imwrite(join(dst_dir, imname), img[:,:,::-1])
                cv2.imwrite(join(dst_dir, maskname), mask[:,:,0])
                with open(join(dst_dir, feature_name), 'wb') as fp:
                    pickle.dump(features, fp)

                pb.update()

        self.toggle_disable_main(enabled=True)

    def initialize_from_root(self):

        subdirs = ['data', 'checkpoints', 'generated']
        for subdir in subdirs:
            if not isdir(join(self.root_dir, subdir)):
                makedirs(join(self.root_dir, subdir))

    def create_image_iterator(self, buffer_size=2):
        while True:
            iter = self.netG.get_images(buffer_size)
            for img, features in iter:
                if self.solver.is_trained:
                    mask, probs = self.solver.predict(features)
                    mask = mask[0].astype(np.uint8)
                    probs = probs[0].astype(np.float32)
                else:
                    mask = None
                    probs = None
                yield img, mask, features, probs

    def save_current_results(self):
        if self.buffer_img is not None:

            self.prepare_drawed_mask()

            image_id = random.randint(0, 1000000)
            dst_dir = join(self.root_dir, 'data')
            mask_name = f'mask_{image_id:06d}.png'
            img_name = f'img_{image_id:06d}.jpg'
            feature_name = f'feat_{image_id:06d}.pickle'

            self.buffer_img.save(join(dst_dir, mask_name))
            Image.fromarray(self.img).save(join(dst_dir, img_name))
            with open(join(dst_dir, feature_name), 'wb') as fp:
                pickle.dump(self.features, fp)

    def next_image(self):

        img_orig, mask, features, _ = next(self.image_iterator)

        img = img_orig
        if mask is not None:
            # mask[:,:,0] = morph_mask(mask[:,:,0])
            img = get_draw_mask(img_orig, mask[:,:,0], alpha=0.5, color_map=None, skip_background=True)
            img = img.astype(np.uint8)

        self.img_orig = img_orig
        self.img = img
        self.features = features

        self.img_frame = ImageTk.PhotoImage(Image.fromarray(img))
        self.can.config(bg='#000000', width=self.img_frame.width(), height=self.img_frame.height())
        self.can.create_image(0, 0, image=self.img_frame, anchor=tk.NW)

        self.buffer_img = Image.new('RGB', (self.img_frame.width(), self.img_frame.height()), (0,0,0))
        self.has_changes = False
        self.history = []