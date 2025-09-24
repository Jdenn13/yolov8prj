import os
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ============ CẤU HÌNH =============
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
ANNOT_DIR = "data/check"
CONF_THRESH = 0.7       
YOLO_CONF_REPLACE = 0.6
CLIP_MIN_PROB = 0.7     

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ LOAD MODEL ============
yolo_model = YOLO("yolov8n.pt").to(DEVICE)  
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ============ TẠO THƯ MỤC ============
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

# ============ LABEL MAP (CHỈ TRONG 1 LẦN CHẠY) ============
label_map = {}

# ============ HỖ TRỢ VẼ BOX ============
drawing = False
ix, iy = -1, -1
new_boxes = []

def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, new_boxes, temp_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = min(ix,x), min(iy,y), max(ix,x), max(iy,y)
        cv2.rectangle(temp_img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.imshow("REVIEW", temp_img)
        cls_id = input("→ Nhập ID cho box thủ công: ")
        if cls_id.strip().isdigit():
            new_boxes.append((int(cls_id), x1,y1,x2,y2))

# ============ HÀM CLIP PHÂN LOẠI ============
def classify_with_clip(image_crop):
    class_names = list(yolo_model.names.values())
    inputs = clip_processor(text=class_names, images=image_crop, return_tensors="pt", padding=True).to(DEVICE)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
    top_idx = int(probs.argmax())
    top_prob = probs[top_idx]
    return top_idx, top_prob

# ============ HÀM CLIP XÁC NHẬN ============
def verify_with_clip(image_crop, label_text):
    inputs = clip_processor(text=[label_text], images=image_crop, return_tensors="pt", padding=True).to(DEVICE)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
    return probs[0] >= CLIP_MIN_PROB

# ============ XỬ LÝ 1 ẢNH ============
def process_image(img_path):
    global temp_img, new_boxes
    filename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    results = yolo_model(img_path)[0]
    label_file = os.path.join(LABEL_DIR, filename + ".txt")
    has_valid_box = False
    new_boxes = []

    with open(label_file, "w") as f:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]

            
            if cls_name.lower() == "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            label_source, label_conf = "YOLO", conf
            if conf < YOLO_CONF_REPLACE:
                cls_id, clip_conf = classify_with_clip(crop_pil)
                if clip_conf < CLIP_MIN_PROB: continue
                cls_name = yolo_model.names[cls_id]
                label_source, label_conf = "CLIP", clip_conf
            else:
                cls_name = yolo_model.names[cls_id]
                if not verify_with_clip(crop_pil, cls_name): continue

            if cls_name in label_map:
                cls_id = label_map[cls_name]
                print(f"[AUTO] Ghi nhớ: {cls_name} → ID={cls_id}")
            else:
                cv2.imshow("CROP", crop)
                cv2.waitKey(1)
                user_input = input(f"→ Nhập ID cho '{cls_name}' (Enter giữ nguyên, q bỏ box): ")
                if user_input.strip().lower() == "q":
                    cv2.destroyWindow("CROP")
                    continue
                elif user_input.strip().isdigit():
                    cls_id = int(user_input.strip())
                    label_map[cls_name] = cls_id
                    print(f"[+] Đã lưu ánh xạ tạm: {cls_name} → ID={cls_id}")
                cv2.destroyWindow("CROP")

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
            has_valid_box = True
            label = f"{cls_name} ({label_source} {label_conf:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ============ Chế độ chỉnh sửa ============
    if has_valid_box:
        temp_img = img.copy()
        cv2.imshow("REVIEW", temp_img)
        cv2.setMouseCallback("REVIEW", draw_box)

        print(f"→ [{filename}] Ấn [y] lưu, [n] bỏ, [d] thêm box thủ công")
        while True:
            key = cv2.waitKey(0)
            if key == ord("y"):
                with open(label_file, "a") as f:
                    for cls_id, x1,y1,x2,y2 in new_boxes:
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
                        cv2.rectangle(temp_img, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.imwrite(os.path.join(ANNOT_DIR, filename + ".jpg"), temp_img)
                print(f"[✓] Đã lưu ảnh và nhãn: {filename}")
                break
            elif key == ord("n"):
                os.remove(label_file)
                print(f"[×] Bỏ qua: {filename}")
                break
            elif key == ord("d"):
                print("→ Dùng chuột để vẽ box, thả chuột để nhập ID.")
        cv2.destroyAllWindows()
    else:
        os.remove(label_file)
        print(f"[×] Không có box hợp lệ: {filename}")

# ============ CHẠY TRÊN THƯ MỤC ============
def main():
    for img_file in os.listdir(IMAGE_DIR):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            process_image(os.path.join(IMAGE_DIR, img_file))
    print("\n[!] Hoàn tất 1 bộ data. Reset lại ID mapping.")
    label_map.clear()

if __name__ == "__main__":
    main()
