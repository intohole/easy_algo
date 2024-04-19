def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"开始执行 {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} 执行完成")
        return result

    return wrapper
