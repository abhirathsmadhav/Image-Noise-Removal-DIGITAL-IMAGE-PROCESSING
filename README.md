# 📸 Image Noise Removal Using Digital Image Processing

## 🎓 FDIP Mini Project
**Team members:**
- 02: ABHIRATH S MADHAV
- 03: ADHILA P M
- 11: ANUAYA P S
- 12: ARCHANA C K

---

## 📖 The Big Picture: What is this project?
Imagine you take a photo at night, and it comes out super grainy. Or imagine an old TV losing signal and showing static. That "grain" and "static" is what we call **Image Noise**.

This project is an interactive, modern application that lets you:
1. **Break it:** Intentionally add digital "noise" (grain/static) to perfect images to simulate real-world camera issues.
2. **Fix it:** Run different mathematical **Filters** to magically clean and restore the image.
3. **Grade it:** Automatically grade how good the filter did using a mathematical score, so we don't just rely on our human eyes!

---

## 🛠️ How Do We Create The Problem? (Types of Noise)
Before we can fix a problem, we need to create it! We simulate two main types of camera errors:

### 1. Gaussian Noise (The "Grainy" Photo)
- **What it is:** This looks like TV static or the heavy grain you see when you take a photo in a very dark room with a bad smartphone camera.
- **How we do it:** We use a mathematical bell curve to randomly make every single pixel slightly brighter or slightly darker.

### 2. Salt & Pepper Noise (The "Dusty" Photo)
- **What it is:** This looks like someone literally sprinkled black pepper and white salt all over your photo.
- **How we do it:** We randomly select a few pixels on the screen and forcefully turn them completely black (`0` = pepper) or completely white (`255` = salt). This simulates dead pixels on a camera sensor.

---

## 🧹 How Do We Solve It? (The Filters)
To clean the photo, we slide a small invisible box (called a "Kernel", usually 5x5 pixels) across the entire image. We look at the pixels inside the box to calculate a fixed color for the pixel in the center.

### 1. Mean Filter (The Simple Blur)
- **How it works:** It adds up the colors of all neighbors inside the box and takes the average.
- **The Result:** It's very fast, but it makes the whole image look blurry. It's **terrible** at removing Salt & Pepper noise, because mixing a black dot with normal pixels just creates an ugly grey smudge!

### 2. Median Filter (The Dot Remover)
- **How it works:** It lines up all the neighbor colors from darkest to brightest, and picks the exact middle (median) color.
- **The Result:** It is **incredible** at removing Salt & Pepper noise! Because black (0) and white (255) are at the extreme ends of the line, they never get picked as the middle value. The dust dot vanishes completely!

### 3. Gaussian Filter (The Smart Blur)
- **How it works:** It's like the Mean filter, but it cares *more* about the pixels closest to the center and *less* about pixels further away.
- **The Result:** It smooths out Gaussian grain much better than the Mean filter, while trying to keep the edges of objects slightly sharper.

### 4. Non-Local Means (The Advanced Filter)
- **How it works:** Instead of just looking at immediate neighbors, it scans the *entire* image for similar-looking patches and averages them together.
- **The Result:** It is the ultimate filter. It removes grain perfectly while keeping edges and textures completely sharp.

---

## 📐 How Do We Grade The Result? (The Metrics)
If two filters both look "okay" to our eyes, how do we prove which one is actually better? We use math to compare the *Filtered Image* back to the *Original Perfect Image*.

### 1. MSE (Mean Squared Error)
We subtract the color of the filtered pixel from the original pixel, square the difference, and average it over the whole image.
- **Goal:** We want this number to be as close to **0** as possible (0 means the images are identical).

### 2. PSNR (Peak Signal-to-Noise Ratio)
This measures how strong the real image (Signal) is compared to the remaining noise (Error). 
- **Goal:** We want this number to be **High**! Anything above **30 dB** means the filter did an excellent job.

---

## 🚀 How to Run the Project
This project comes with a beautiful, modern **Dark-Mode GUI (Graphical User Interface)** so you can present it live in front of the class!

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the App:
```bash
python gui.py
```

### 🎮 What you can do in the App during your presentation:
- **Tab 1 - Image Processing:** Click "Load Image" to load a test photo (like `lena.jpg` from the `data/` folder). Use the sliders to add noise in real-time, and click different filters to watch them clean the image instantly. Point out how the PSNR score updates live at the bottom!
- **Tab 2 - Metrics History:** Every time you try a new filter, the app logs the mathematical result here in a neat table. You can use this to prove to the class which filter scored the highest.
- **Tab 3 - Performance Graphs:** Click "Generate Live Graphs" to automatically test all 4 filters and draw beautiful Bar and Line charts comparing their MSE and PSNR scores!
