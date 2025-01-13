use venv: python -m venv .venv ( read https://fastapi.tiangolo.com/virtual-environments/ )

tạo bảng user CREATE TABLE users (
    id SERIAL PRIMARY KEY,             -- Tạo id tự động tăng
    account VARCHAR(255) UNIQUE NOT NULL, -- Trường account, duy nhất và không được null
    password VARCHAR(255) NOT NULL,     -- Trường password, không được null
    role VARCHAR(50) NOT NULL,          -- Trường role, ví dụ 'user' hoặc 'admin'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Thời gian tạo tài khoản
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- Thời gian cập nhật tài khoản
);

