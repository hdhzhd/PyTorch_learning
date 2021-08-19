# coding ：UTF-8
# 文件功能： 代码实现class基本功能
# 开发人员： dpp
# 开发时间： 2021/8/11 11:14 上午
# 文件名称： class_use.py
# 开发工具： PyCharm

class Person:
    def __init__(self, name):
        print("__init__ " + name)

    def __call__(self, name):
        print("__call__ " + "hello " + name)

    def hello(self, name):
        print("hello " + name)


person = Person("first name")    # 对象创建时初始化，自动调用__init__函数
person.__call__("zhangsan")   # 对象显示地调用__call__函数
person("zhangsan")     # 对象直接调用__call__函数，调用效果与显示调用__call__一样
person.hello("lisi")    # 对象调用hello方法
