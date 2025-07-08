import os, glob, re

INPUT_DIR  = r".\whisper\test\testtxtM"    
OUTPUT_DIR = r".\whisper\test\testtxtM\cleaned"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

char_map = {
    '{': 'š',  # { → š
    '~': 'č',  # ~ → č
    '^': 'ć',  # ^ → ć
    '`': 'ž',  # ` → ž
    '}': 'đ',  # } → đ
}

char_map.update({k.upper(): v.upper() for k, v in char_map.items()})
translator = str.maketrans(char_map)

for path in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
    with open(path, encoding="utf-8") as f:
        text = f.read()
        
    text = re.sub(r"<[^>]*>", "", text)
    text = text.translate(translator)
    
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

print("Gotovo čišćenje!") 
