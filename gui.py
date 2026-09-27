import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import our existing backend functions
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise
from src.filters import apply_mean_filter, apply_median_filter, apply_gaussian_filter, apply_nlm_filter
from src.metrics import calculate_mse, calculate_psnr

# Set Modern Appearance
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class ModernImageNoiseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("FDIP Mini Project - Advanced Image Noise Removal")
        self.geometry("1300x750")
        
        # State
        self.original_image = None
        self.noisy_image = None
        self.filtered_image = None
        
        # Cached states to avoid redundant recalculation
        self._last_noise_state = None
        self._last_filter_state = None
        
        # Display target dynamic size (starts at 350x350)
        self.display_size = (350, 350)
        
        # Debounce timer for resizing
        self._resize_timer = None
        
        # Handle graceful shutdown
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()

    def on_closing(self):
        if self._resize_timer is not None:
            self.after_cancel(self._resize_timer)
            self._resize_timer = None
        self.quit()
        self.destroy()

    def setup_ui(self):
        # Create Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_image = self.tabview.add("Image Processing")
        self.tab_metrics = self.tabview.add("Metrics History")
        self.tab_graphs = self.tabview.add("Performance Graphs")
        
        # Configure Grid for tab_image
        self.tab_image.grid_columnconfigure(0, weight=1)
        self.tab_image.grid_columnconfigure(1, weight=1)
        self.tab_image.grid_columnconfigure(2, weight=1)
        self.tab_image.grid_rowconfigure(0, weight=1) # Images
        self.tab_image.grid_rowconfigure(1, weight=0) # Controls
        
        # --- Image Panels (Top) ---
        # 1. Original
        orig_frame = ctk.CTkFrame(self.tab_image)
        orig_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(orig_frame, text="Original Image", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.lbl_orig = ctk.CTkLabel(orig_frame, text="No Image Loaded", width=350, height=350, fg_color="gray20", corner_radius=10)
        self.lbl_orig.pack(pady=10, padx=10, expand=True, fill="both")
        
        # 2. Noisy
        noisy_frame = ctk.CTkFrame(self.tab_image)
        noisy_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.lbl_noisy_title = ctk.CTkLabel(noisy_frame, text="Noisy Image", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_noisy_title.pack(pady=5)
        self.lbl_noisy = ctk.CTkLabel(noisy_frame, text="Waiting...", width=350, height=350, fg_color="gray20", corner_radius=10)
        self.lbl_noisy.pack(pady=10, padx=10, expand=True, fill="both")
        
        # 3. Filtered
        filt_frame = ctk.CTkFrame(self.tab_image)
        filt_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.lbl_filt_title = ctk.CTkLabel(filt_frame, text="Filtered Image", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_filt_title.pack(pady=5)
        self.lbl_filt = ctk.CTkLabel(filt_frame, text="Waiting...", width=350, height=350, fg_color="gray20", corner_radius=10)
        self.lbl_filt.pack(pady=10, padx=10, expand=True, fill="both")
        
        # Bind resize event to one of the frames to track window size changes
        orig_frame.bind("<Configure>", self.schedule_resize)
        
        self.lbl_metrics = ctk.CTkLabel(filt_frame, text="MSE: --  |  PSNR: -- dB", font=ctk.CTkFont(size=16, weight="bold"), text_color="#1f6aa5")
        self.lbl_metrics.pack(pady=5)

        # --- Controls (Bottom) ---
        control_frame = ctk.CTkFrame(self.tab_image)
        control_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Left: Load & Base Controls
        left_ctrl = ctk.CTkFrame(control_frame, fg_color="transparent")
        left_ctrl.pack(side="left", padx=20, pady=10)
        
        self.btn_load = ctk.CTkButton(left_ctrl, text="Load Image", command=self.load_image, font=ctk.CTkFont(weight="bold"))
        self.btn_load.pack(pady=10)
        
        # Middle: Noise Config
        noise_ctrl = ctk.CTkFrame(control_frame)
        noise_ctrl.pack(side="left", padx=20, pady=10, fill="y")
        ctk.CTkLabel(noise_ctrl, text="1. Noise Settings", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.noise_var = ctk.StringVar(value="None")
        ctk.CTkRadioButton(noise_ctrl, text="None", variable=self.noise_var, value="None", command=self.trigger_update).pack(side="left", padx=10)
        ctk.CTkRadioButton(noise_ctrl, text="Gaussian", variable=self.noise_var, value="Gaussian", command=self.trigger_update).pack(side="left", padx=10)
        ctk.CTkRadioButton(noise_ctrl, text="Salt & Pepper", variable=self.noise_var, value="SaltPepper", command=self.trigger_update).pack(side="left", padx=10)
        
        slider_frame1 = ctk.CTkFrame(noise_ctrl, fg_color="transparent")
        slider_frame1.pack(side="bottom", fill="x", padx=10, pady=10)
        ctk.CTkLabel(slider_frame1, text="Intensity:").pack(side="left", padx=5)
        self.slider_noise = ctk.CTkSlider(slider_frame1, from_=0.01, to=0.2, command=self.trigger_update)
        self.slider_noise.set(0.05)
        self.slider_noise.pack(side="left", fill="x", expand=True)

        # Right: Filter Config
        filter_ctrl = ctk.CTkFrame(control_frame)
        filter_ctrl.pack(side="left", padx=20, pady=10, fill="both", expand=True)
        ctk.CTkLabel(filter_ctrl, text="2. Filter Settings", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.filter_var = ctk.StringVar(value="None")
        
        # Use a sub-frame and grid to organize the 4 buttons neatly in 2 rows
        filter_rb_frame = ctk.CTkFrame(filter_ctrl, fg_color="transparent")
        filter_rb_frame.pack(pady=0)
        
        ctk.CTkRadioButton(filter_rb_frame, text="None", variable=self.filter_var, value="None", command=self.trigger_update).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkRadioButton(filter_rb_frame, text="Mean", variable=self.filter_var, value="Mean", command=self.trigger_update).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ctk.CTkRadioButton(filter_rb_frame, text="Median", variable=self.filter_var, value="Median", command=self.trigger_update).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkRadioButton(filter_rb_frame, text="Gaussian", variable=self.filter_var, value="Gaussian", command=self.trigger_update).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ctk.CTkRadioButton(filter_rb_frame, text="NLM (Advanced)", variable=self.filter_var, value="NLM", command=self.trigger_update).grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        slider_frame2 = ctk.CTkFrame(filter_ctrl, fg_color="transparent")
        slider_frame2.pack(side="bottom", fill="x", padx=10, pady=10)
        ctk.CTkLabel(slider_frame2, text="Strength/Size:").pack(side="left", padx=5)
        self.slider_kernel = ctk.CTkSlider(slider_frame2, from_=3, to=15, number_of_steps=6, command=self.trigger_update)
        self.slider_kernel.set(5)
        self.slider_kernel.pack(side="left", fill="x", expand=True)

        # Setup Tabs
        self.setup_metrics_tab()
        self.setup_graphs_tab()

    def setup_graphs_tab(self):
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_rowconfigure(1, weight=1)
        
        control_bar = ctk.CTkFrame(self.tab_graphs, fg_color="transparent")
        control_bar.grid(row=0, column=0, pady=10)
        
        ctk.CTkLabel(control_bar, text="Evaluate current noisy image against all filters:", font=ctk.CTkFont(size=14)).pack(side="left", padx=10)
        ctk.CTkButton(control_bar, text="Generate Live Graphs", command=self.generate_graphs, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        
        self.graph_frame = ctk.CTkFrame(self.tab_graphs)
        self.graph_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.canvas_widget = None

    def generate_graphs(self):
        if self.original_image is None or self.noisy_image is None:
            messagebox.showwarning("Warning", "Please load an image and add noise first on the Image Processing tab!")
            return
            
        current_noise = self.noise_var.get()
        if current_noise == "None":
            messagebox.showwarning("Warning", "Please add Gaussian or Salt & Pepper noise to evaluate filters!")
            return
            
        # Run benchmark
        mean_img = apply_mean_filter(self.noisy_image, kernel_size=5)
        median_img = apply_median_filter(self.noisy_image, kernel_size=5)
        gaussian_img = apply_gaussian_filter(self.noisy_image, kernel_size=5, sigma=1.5)
        nlm_img = apply_nlm_filter(self.noisy_image, h=10)
        
        filters = ["Mean", "Median", "Gaussian", "NLM"]
        images = [mean_img, median_img, gaussian_img, nlm_img]
        
        mses = [calculate_mse(self.original_image, img) for img in images]
        psnrs = [calculate_psnr(self.original_image, img) for img in images]
        
        # Clear old graph
        if self.canvas_widget:
            self.canvas_widget.destroy()
            
        # Generate Matplotlib Figure
        fig, ax1 = plt.subplots(figsize=(8, 5))
        # Ensure dark theme compatibility (matplotlib defaults are light)
        fig.patch.set_facecolor('#2b2b2b')
        ax1.set_facecolor('#2b2b2b')
        ax1.tick_params(colors='white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        
        color = '#ff6666'
        ax1.set_xlabel('Filter Type')
        ax1.set_ylabel('MSE (Lower is Better)', color=color)
        ax1.bar(filters, mses, color=color, alpha=0.8, width=0.4, label="MSE")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = '#66b3ff'
        ax2.set_ylabel('PSNR in dB (Higher is Better)', color=color)  
        ax2.plot(filters, psnrs, color=color, marker='o', linewidth=3, markersize=10, label="PSNR")
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f"Filter Performance on {current_noise} Noise", color='white', pad=20)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def setup_metrics_tab(self):
        self.tab_metrics.grid_columnconfigure(0, weight=1)
        self.tab_metrics.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(self.tab_metrics, text="Historical Metrics Log", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, pady=10)
        
        # Create a scrollable frame for the table
        self.metrics_scroll = ctk.CTkScrollableFrame(self.tab_metrics)
        self.metrics_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        # Table Headers
        headers = ["Time", "Noise Type", "Noise Int.", "Filter Type", "Kernel/Strength", "MSE", "PSNR (dB)"]
        for col, h in enumerate(headers):
            self.metrics_scroll.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(self.metrics_scroll, text=h, font=ctk.CTkFont(weight="bold")).grid(row=0, column=col, padx=5, pady=5)
            
        self.metrics_row_count = 1
        self.metrics_history = []
        
        # Button to clear history
        ctk.CTkButton(self.tab_metrics, text="Clear History", command=self.clear_metrics).grid(row=2, column=0, pady=10)

    def log_metric(self, n_type, n_int, f_type, f_param, mse, psnr):
        t = time.strftime("%H:%M:%S")
        data = [t, n_type, f"{n_int:.3f}", f_type, str(f_param), f"{mse:.2f}", f"{psnr:.2f}"]
        
        # Avoid duplicate consecutive logs
        if self.metrics_history and self.metrics_history[-1][1:] == data[1:]:
            return
            
        self.metrics_history.append(data)
        
        for col, val in enumerate(data):
            ctk.CTkLabel(self.metrics_scroll, text=val).grid(row=self.metrics_row_count, column=col, padx=5, pady=2)
            
        self.metrics_row_count += 1
        
    def clear_metrics(self):
        for widget in self.metrics_scroll.winfo_children():
            # Keep headers (row=0)
            if int(widget.grid_info()['row']) > 0:
                widget.destroy()
        self.metrics_row_count = 1
        self.metrics_history.clear()

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not filepath: return
            
        self.original_image = cv2.imread(filepath)
        if self.original_image is None:
            messagebox.showerror("Error", "Could not read the image.")
            return
            
        # Reset UI
        self.noise_var.set("None")
        self.filter_var.set("None")
        self.slider_noise.set(0.05)
        self.slider_kernel.set(5)
        
        self.update_image_label(self.original_image, self.lbl_orig)
        self.trigger_update()

    def _cv_to_ctk(self, cv_img):
        """Converts OpenCV BGR image to CustomTkinter Image"""
        if len(cv_img.shape) == 3:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            
        pil_img = Image.fromarray(rgb_img)
        # Using CTkImage to handle high DPI scaling cleanly
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=self.display_size)

    def update_image_label(self, cv_img, ctk_label):
        if cv_img is None:
            ctk_label.configure(image=None, text="None")
            return
            
        # Manually resize to fit max bounds for CTkImage size property
        h, w = cv_img.shape[:2]
        scale = min(self.display_size[0]/w, self.display_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = self._cv_to_ctk(cv_img)
        img.configure(size=(new_w, new_h))
        
        ctk_label.configure(image=img, text="")
        ctk_label.image = img # Prevent GC

    def schedule_resize(self, event):
        # We only care about resizing the image frames
        if event.width < 100 or event.height < 100: return
        
        # Determine new available space (padding taken into account)
        new_w = event.width - 20
        new_h = event.height - 40 # Account for title label
        
        if (new_w, new_h) != self.display_size:
            self.display_size = (new_w, new_h)
            if self._resize_timer is not None:
                self.after_cancel(self._resize_timer)
            self._resize_timer = self.after(150, self.refresh_images_size)
            
    def refresh_images_size(self):
        if self.original_image is not None:
            self.update_image_label(self.original_image, self.lbl_orig)
        if self.noisy_image is not None:
            self.update_image_label(self.noisy_image, self.lbl_noisy)
        if self.filtered_image is not None:
            self.update_image_label(self.filtered_image, self.lbl_filt)

    def trigger_update(self, *args):
        """Called whenever a slider or radio button changes to trigger live preview."""
        if self.original_image is None: return
        
        # Debounce/Delay mechanism isn't strictly necessary if it's fast enough,
        # but we optimize by caching states.
        
        # 1. Process Noise
        current_noise = self.noise_var.get()
        intensity = self.slider_noise.get()
        noise_state = (current_noise, intensity)
        
        if self.noisy_image is None or self._last_noise_state != noise_state:
            self._last_noise_state = noise_state
            
            if current_noise == "Gaussian":
                self.noisy_image = add_gaussian_noise(self.original_image, var=intensity)
                self.lbl_noisy_title.configure(text=f"Noisy Image (Gaussian, var={intensity:.3f})")
            elif current_noise == "SaltPepper":
                self.noisy_image = add_salt_and_pepper_noise(self.original_image, amount=intensity)
                self.lbl_noisy_title.configure(text=f"Noisy Image (S&P, amt={intensity:.3f})")
            else:
                self.noisy_image = self.original_image.copy()
                self.lbl_noisy_title.configure(text="Noisy Image (None)")
                
            self.update_image_label(self.noisy_image, self.lbl_noisy)

        # 2. Process Filter
        current_filter = self.filter_var.get()
        # Ensure kernel is odd
        kernel = int(self.slider_kernel.get())
        if kernel % 2 == 0: kernel += 1 
        
        filter_state = (current_filter, kernel, noise_state) # if noise changes, filter must recalculate
        
        if self.filtered_image is None or self._last_filter_state != filter_state:
            self._last_filter_state = filter_state
            
            if current_filter == "Mean":
                self.filtered_image = apply_mean_filter(self.noisy_image, kernel_size=kernel)
                self.lbl_filt_title.configure(text=f"Filtered Image (Mean, k={kernel})")
            elif current_filter == "Median":
                self.filtered_image = apply_median_filter(self.noisy_image, kernel_size=kernel)
                self.lbl_filt_title.configure(text=f"Filtered Image (Median, k={kernel})")
            elif current_filter == "Gaussian":
                # Dynamic sigma based on kernel size
                sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
                self.filtered_image = apply_gaussian_filter(self.noisy_image, kernel_size=kernel, sigma=sigma)
                self.lbl_filt_title.configure(text=f"Filtered Image (Gaussian, k={kernel})")
            elif current_filter == "NLM":
                # Use the slider value as the filter strength 'h' (usually 3 to 15 is good)
                strength = kernel 
                self.filtered_image = apply_nlm_filter(self.noisy_image, h=strength)
                self.lbl_filt_title.configure(text=f"Filtered Image (NLM, h={strength})")
            else:
                self.filtered_image = self.noisy_image.copy()
                self.lbl_filt_title.configure(text="Filtered Image (None)")
                
            self.update_image_label(self.filtered_image, self.lbl_filt)
            
            # Update Metrics (Comparing original clean to final filtered)
            mse = calculate_mse(self.original_image, self.filtered_image)
            psnr = calculate_psnr(self.original_image, self.filtered_image)
            
            # If no noise and no filter, PSNR is infinity. Handle it nicely.
            if mse == 0:
                self.lbl_metrics.configure(text="MSE: 0.00  |  PSNR: ∞ dB", text_color="green")
            else:
                # Color code metrics (higher PSNR is better)
                color = "#00cc66" if psnr > 30 else ("#ffcc00" if psnr > 22 else "#ff3333")
                self.lbl_metrics.configure(text=f"MSE: {mse:.2f}  |  PSNR: {psnr:.2f} dB", text_color=color)
                
            # Record to Metrics Tab
            if current_filter != "None" or current_noise != "None":
                param = kernel if current_filter != "NLM" else strength
                self.log_metric(current_noise, intensity, current_filter, param, mse, psnr)

if __name__ == "__main__":
    app = ModernImageNoiseApp()
    app.mainloop()
