use venv: python -m venv .venv ( read https://fastapi.tiangolo.com/virtual-environments/ )

## install tesseract-ocr

sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-vie # Cài đặt Tesseract và ngôn ngữ tiếng Việt (tùy chọn)
tesseract --version # Kiểm tra cài đặt

## install Poppler

sudo apt install poppler-utils
pdftoppm -v # Kiểm tra cài đặt

## check PATH

which tesseract # Kiểm tra Tesseract
which pdftoppm # Kiểm tra Poppler
