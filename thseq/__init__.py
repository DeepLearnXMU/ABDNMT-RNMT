# import math
# import matplotlib.pyplot as plt
#
# ax = plt.subplot()
#
# warmup = 500
# warmup_init = 0
# decay_start = 6e5
# decay_end = 12e5
# min_ = 0.5
# max_ = 1e8
# num_splits = 2
#
# splits = num_splits
# warmup_end = warmup * splits
# decay_start = max(warmup_end + 1.0, decay_start / splits)
# peak = 1.0 * splits
# decay_end = max(decay_start + 1.0, decay_end / splits)
#
# print(warmup_end,decay_start,decay_end)
# class BaseLRSchedule(object):
#     def __init__(self,name=None):
#         super().__init__()
#         self.name = name or str(self.__class__)
#
#     def get_value(self, step):
#         raise NotImplementedError
#
#
# class PolynomialLRSchedule(BaseLRSchedule):
#     def __init__(self, x0, y0, x1, y1, power=1,name=None):
#         super().__init__(name)
#         assert x0 < x1, '%s must be < %s' % (x0, x1)
#
#         def polynomial(x):
#             fx = ((x - x0) / (x1 - x0)) ** power
#             y = y0 + fx * (y1 - y0)
#             if x < x0:
#                 return y0
#             elif x >= x1:
#                 return y1
#             else:
#                 return y
#
#         self.polynomial = polynomial
#
#     def get_value(self, step):
#         return self.polynomial(step)
#
#
# class LinearLRSchedule(PolynomialLRSchedule):
#     def __init__(self, x0, y0, x1, y1,name=None):
#         super().__init__(x0, y0, x1, y1, power=1,name=name)
#
#
# class ExponentialLRSchedule(BaseLRSchedule):
#     def __init__(self, x0, y0, x1, y1,name=None):
#         super().__init__(name)
#         self.linear = LinearLRSchedule(x0, math.log(y0), x1, math.log(y1))
#
#         def exp(x):
#             return math.exp(self.linear.get_value(x))
#
#         self.exp = exp
#
#     def get_value(self, step):
#         return self.exp(step)
#
#
# class CombinedMinimumLRSchedule(BaseLRSchedule):
#     def __init__(self, schedulers,name=None):
#         super().__init__(name)
#
#         def combined(x):
#             ys = [s.get_value(x) for s in schedulers]
#             return min(ys)
#
#         self.combined = combined
#
#     def get_value(self, step):
#         return self.combined(step)
#
#
# decay = CombinedMinimumLRSchedule([
#     LinearLRSchedule(0, 0, warmup_end, peak),
#     LinearLRSchedule(warmup_end, peak, decay_start, peak),
#     ExponentialLRSchedule(decay_start, peak, decay_end, min_),
#     # LinearLRSchedule(0, max_, decay_end, max_)
# ])
#
# # print(decay.get_value(300000))
# # print(decay.get_value(300000+1))
# #
# import numpy
# # x=numpy.arange(decay_end*2)+1
# x=numpy.arange(warmup_end)+1
# y=numpy.array([decay.get_value(i) for i in x])*0.0001
#
# ax.plot(x/x.max(),y)
# ax.set_xlim([0.0,1.0])
# plt.show()
