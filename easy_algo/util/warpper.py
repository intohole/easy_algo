def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"start exec {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed successfully")
        return result

    return wrapper
