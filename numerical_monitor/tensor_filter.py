"""可扩展的张量过滤链，独立于训练框架。

保持 NumericalAnalyzer 的 FilterRegistry.get_chain(group).should_skip(...) 调用方式，
由业务显式配置 GT 等排除规则，避免监控器根据张量形状或名称猜测真值语义。
"""

from collections import defaultdict


class FilterChain:
    """依次执行张量过滤规则；任一规则返回 True，即跳过该张量的数值统计。"""

    def __init__(self, filters=()):
        self.filters = list(filters)

    def should_skip(self, position, tensor, module, identity):
        """兼容带 should_skip 方法的过滤对象，以及相同签名的普通函数。"""
        for predicate in self.filters:
            if hasattr(predicate, "should_skip"):
                skip = predicate.should_skip(position, tensor, module, identity)
            else:
                skip = predicate(position, tensor, module, identity)
            if skip:
                return True
        return False


class FilterRegistry:
    """进程内过滤器注册表；group=None 表示对所有业务组生效的全局规则。"""

    _filters = defaultdict(list)

    @classmethod
    def register(cls, predicate, group=None):
        """注册组专属或全局规则，返回原规则以支持装饰器式使用。"""
        cls._filters[group].append(predicate)
        return predicate

    @classmethod
    def get_chain(cls, group):
        """获取当前 group 的张量过滤链（全局默认 + group 专属）。"""
        return FilterChain(cls._filters[None] + cls._filters[group])

    @classmethod
    def clear(cls):
        """清空当前进程规则，供独立实验或测试重置过滤配置。"""
        cls._filters.clear()
