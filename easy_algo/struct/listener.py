# 监听者接口
class ListenerInterface:
    def notify(self, event, *args, **kwargs):
        """处理逻辑"""
        pass


# 被监听者类
class Subject:
    def __init__(self):
        self._listeners = []

    def add_listener(self, listener):
        """ 注册一个监听者 """
        self._listeners.append(listener)

    def remove_listener(self, listener):
        """ 移除一个监听者 """
        self._listeners.remove(listener)

    def notify_listeners(self, event, *args, **kwargs):
        """ 通知所有注册的监听者 """
        for listener in self._listeners:
            listener.update(event, *args, **kwargs)
