import numpy as np
import matplotlib.pyplot as plt

########## numpy math func ##########
### 单位转换
print("="*30)
theta_deg = [0,30,60,90,120,150,180]
print("theta degree: ",theta_deg)
theta_rad = np.deg2rad(theta_deg)   # 角度单位转弧度单位
print("theta rad: ", theta_rad)    # 打印输出
theta_rad_around = np.around(theta_rad,decimals = 2)    # 保留指定小数位数
print(theta_rad_around) # 打印输出
### 三角函数
print("="*30)
theta_sin = np.sin(theta_rad)   # sin 正弦函数 输入单位：弧度
print(theta_sin)    # 打印输出
theta_cos = np.cos(theta_rad)   # cos
print(theta_cos)
theta_tan = np.tan(theta_rad)   # tan
print(theta_tan)
theta_arcsin = np.arcsin(theta_sin) # arcsin
print(theta_arcsin)
theta_arccos = np.arccos(theta_cos) # arccos
print(theta_arccos)
theta_arctan = np.arctan(theta_tan) # arctan

########## create array ##########
########## 创建数组 ##########
print("="*30)
array_1 = np.array([1,2,3])
array_2 = np.zeros(2)
array_3 = np.ones(2)
array_4 = np.empty(2)
array_5 = np.arange(4)
array_6 = np.arange(2,9,2)
array_7 = np.linspace(0,10,num=5)
array_8 = np.ones(2,dtype=np.int64)
print("array_1: ",array_1)
print("array_2: ",array_2)
print("array_3: ",array_3)
print("array_4: ",array_4)
print("array_5: ",array_5)
print("array_6: ",array_6)


########## adding removing sorting elements ##########
########## 添加、移除、排序数组元素 ##########
print("="*30)
array_9 = np.array([2,1,5,3,7,4,6,8])
array_9_sort = np.sort(array_9)
print("array :", array_9)
print("array sort: ", array_9_sort)
array_10 = np.array([1,2,3,4])
array_11 = np.array([5,6,7,8])
array_10_11_concatente = np.concatenate((array_10,array_11))
print("array concat 1: ", array_10)
print("array concat 2: ", array_11)
print("array concat: ",array_10_11_concatente)
array_12 = np.array([[1,2],[3,4]])
array_13 = np.array([[5,6]])
array_12_13_concatente = np.concatenate((array_12,array_13),axis=0)
print("array concat 1: ", array_12)
print("array concat 2: ", array_13)
print("array concat: ",array_12_13_concatente)

########## array shape and size ##########
########## 数组维度、大小与形状获取 ##########
print("="*30)
array_14 = np.array([[[0,1,2,3],[4,5,6,7]],
            [[0,1,2,3],[4,5,6,7]],
            [[0,1,2,3],[4,5,6,7]]])
# array_14_dim = np.ndim(array_14)
# array_14_size = np.size(array_14)
# array_14_shape = np.shape(array_14)
array_14_dim = array_14.ndim
array_14_size = array_14.size
array_14_shape = array_14.shape
print("array_14: ",array_14)
print("array_14_dim",array_14_dim)
print("array_14_size",array_14_size)
print("array_14_shape",array_14_shape)

########## reshape array ##########
########## 改变数组形状 ##########
print("="*30)
array_15 = np.arange(6)
array_15_reshape = array_15.reshape(3,2)
array_15_reshape_1 = np.reshape(array_15,newshape=(1,6),order='C')
print("array_15: ",array_15)
print("array_15_reshape: ",array_15_reshape)
print("array_15_reshape_1: ",array_15_reshape_1)
print(array_15.shape)
print(array_15_reshape_1.shape)


########## covert 1D array into 2D array ##########
########## 一维数组转二维数组，添加新轴 ##########
print("="*30)
array_16 = np.array([1,2,3,4,5,6])
array_16_shape = array_16.shape
array_16_newaxis = array_16[np.newaxis,:]   # 转为行向量
array_16_newaxis_shape = array_16_newaxis.shape
array_17_newaxis = array_16[:,np.newaxis]
array_17_newaxis_shape = array_17_newaxis.shape
print("array_16: ",array_16)
print("array_16_shape: ",array_16_shape)
print("array_16_newaxis: ",array_16_newaxis)
print("array_16_newaxis_shape: ",array_16_newaxis_shape)
print("array_17_newaxis: ",array_17_newaxis)
print("array_17_newaxis_shape: ",array_17_newaxis_shape)

array_17 = np.array([1,2,3,4,5,6])
array_17_shape = array_17.shape
array_17_expand_row = np.expand_dims(array_17,axis=1)
array_17_expand_row_shape = array_17_expand_row.shape
array_17_expand_col = np.expand_dims(array_17,axis=0)
array_17_expand_col_shape = array_17_expand_col.shape
print("array_17: ",array_17)
print("array_17_shape: ", array_17_shape)
print("array_17_expand_row: ",array_17_expand_row)
print("array_17_expand_row_shape: ",array_17_expand_row_shape)
print("array_17_expand_col: ",array_17_expand_col)
print("array_17_expand_col_shape: ",array_17_expand_col_shape)


########## indexing and slicing ##########
########## 数组索引与切片 ##########
print("="*30)
array_18 = np.array([1,2,3])
print("array_18 1th elements: ",array_18[0])
print("array_18 1-2 elements: ",array_18[0:2])
print("array_18 1: elements: ",array_18[1:])
print("array_18 -2: elements: ",array_18[-2:])

array_19 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("array_19 < 5: ",array_19[array_19 <5])
array_19_index = (array_19 >= 5 )
print("array_19 >= 5: ",array_19[array_19_index])
print("array_19 %2 =0: ", array_19[array_19%2==0])
print("array_19 > 5 | array_19 == 5: ",array_19[(array_19>5)|(array_19==5)])
print("array bool index: ", (array_19>5)|(array_19==5))

array_19_nonzero = np.nonzero(array_19 < 5)
print("array_19 < 5 row and col index: ",array_19_nonzero)
print("array_19 < 5 nonzero: ",array_19[array_19_nonzero])

array_19_list_zip = list(zip(array_19_nonzero[0],array_19_nonzero[1]))
print(array_19_list_zip)
for index in array_19_list_zip:
    print(index)


########## create array from existing data ##########
########## 从已有数据创建数组 ##########
print("="*30)
array_20 = np.array([1,2,3,4,5,6,7,8,9,10])
print("array_20_array: ",array_20)
array_20_arange = np.arange(1,11)
print("array_20_arange: ",array_20_arange)
array_20_1 = array_20[3:8]
print("array_20_1: ",array_20_1)
array_21 = np.array([[1,1],[2,2]])
array_22 = np.array([[3,3],[4,4]])
array_21_22_vstack = np.vstack((array_21,array_22))
array_21_22_hstack = np.hstack((array_21,array_22))
print("array_21: ",array_21)
print("array_22: ",array_22)
print("array_21_22_vstack: ",array_21_22_vstack)
print("array_21_22_hstack: ",array_21_22_hstack)

array_23  = np.arange(1,25).reshape(2,12)
print("array_23: \n",array_23)
array_23_hsplit = np.hsplit(array_23,3)
print("array_23_hsplit: ",array_23_hsplit)
array_23_hsplit_1 = np.hsplit(array_23,(3,4))
print("array_23_hsplit_1: ",array_23_hsplit_1)

array_24 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
array_24_arange = np.arange(1,13).reshape(3,4)
print("array_24: ",array_24)
print("array_24_arange: ",array_24_arange)
array_24_1 = array_24[0,:]  # 浅拷贝
print("array_24_1: ",array_24_1)
array_24_1[0] = 99
print("array_24: ",array_24)
print("array_24_1: ",array_24_1)
array_24_2 = array_24.copy()    # 深拷贝
array_24_2[0,2] = 99
print("array_24: ",array_24)
print("array_24_2: ",array_24_2)


########## Basic array operations  ##########
########## 基本数组操作 ##########
print("="*30)
array_25 = np.array([1,2])
array_26 = np.ones(2,dtype=int)
array_25add26 = array_25 + array_26
print("add array: ",array_25add26)
array_25sub26 = array_25-array_26
print("sub array: ",array_25sub26)
array_25multi26 = array_25*array_26
print("multi array:", array_25multi26)
array_25div26 = array_25/array_26
print("div array: ",array_25div26)

array_27 = np.array([1,2,3,4])
array_27_sum = array_27.sum()
print("array_27 sum: ",array_27_sum)
array_28 = np.array([[1,1],[2,2]])
array_28_sum_axis0 = array_28.sum(axis=0)
array_28_sum_axis1 = array_28.sum(axis=1)
print("array_28_sum_axis0: ",array_28_sum_axis0)
print("array_28_sum_axis1: ",array_28_sum_axis1)

########## Broadcasting ##########
########## 广播 ##########
print("="*30)
array_29 = np.array([1.0,2.0])
array_29 = array_29*1.6
print("array_29 scalar: ",array_29)

########## More useful array operations ##########
########## 更多针对数组的操作：求极值、中值、求和 标准差等 ##########
print("="*30)
array_30 = np.arange(1,11)
print("array_30: ",array_30)
print("array_30_max: ",array_30.max())
print("array_30_min: ",array_30.min())
print("array_30_sum: ",array_30.sum())

array_31 = np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
              [0.54627315, 0.05093587, 0.40067661, 0.55645993],
              [0.12697628, 0.82485143, 0.26590556, 0.56917101]])
print("array_31: ",array_31)
print("array_31_max: ",array_31.max())
print("array_31_min: ",array_31.min())
print("array_31_min_col: ",array_31.min(axis=0))


########## Creating matrices ##########
########## 创建矩阵 ##########
print("="*30)
array_32 = np.array([[1,2],[3,4],[5,6]])
print("array_32: ",array_32)
print("array_32[0,1]: ",array_32[0,1])
print("array_32[1:3]: ",array_32[1:3])
print("array_32[0:2,0]: ",array_32[0:2,0])

print("array_32_max: ",array_32.max())
print("array_32_min: ",array_32.min())
print("array_32_sum: ",array_32.sum())


array_33 =  np.array([[1, 2], [5, 3], [4, 6]])
print("array_33: ",array_33)
print("array_33_max_axis0",array_33.max(0))
print("array_33_max_axis1",array_33.max(1))

array_34 = np.array([[1, 2], [3, 4]])
array_35 = np.array([[1, 1], [1, 1]])
array_34add35 = array_34 + array_35
print("array_34add35: ",array_34add35)

array_36 = np.array([[1, 2], [3, 4], [5, 6]])
array_37 = np.array([[1,1]])
print("array_36: ",array_36)
print("array_37: ",array_37)
print("array_36add37: ",array_36+array_37)

array_38 = np.ones((4, 3, 2))
print("array_38: ",array_38)

### 1-D
print("array_ones: ",np.ones(3))
print("array_zeros: ",np.zeros(3))
array_39 = np.random.default_rng()
print("array_random: ",array_39.random(3))
### 2-D
print("array_ones: ",np.ones((3,2)))
print("array_zeros: ",np.zeros((3,2)))
array_39 = np.random.default_rng()
print("array_random: ",array_39.random((3,2)))


########## Generating randon numbers ##########
########## 生成随机数 ##########
print("="*30)
print("random intergers: \n",array_39.integers(5,size=(2,4)))


########## How to get unique items and counts  ##########
########## 获取唯一元素 与 元素出现次数统计 ##########
print("="*30)
array_40 = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
array_40_unique = np.unique(array_40)
print("array_40_unique: ", array_40_unique)
array_40_unique,array_40_index = np.unique(array_40,return_index=True)
print("array_40_index: ",array_40_index)
array_40_unique,array_40_count = np.unique(array_40,return_counts=True)
print("array_40_count: ",array_40_count)

array_40_2D = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
array_40_2D_unique = np.unique(array_40_2D)
print("array_40_2D_unique: ",array_40_2D_unique)
array_40_2D_unique_row = np.unique(array_40_2D,axis=0)
print("array_40_2D_unique_row: \n",array_40_2D_unique_row)
array_40_2D_unique, array_40_2D_index,array_40_2D_count = np.unique(array_40_2D,axis=0,return_index=True,return_counts=True)
print("array_40_2D_unique: ",array_40_2D_unique)
print("array_40_2D_index: ",array_40_2D_index)
print("array_40_2D_count: ",array_40_2D_count)


########## Transposing and reshaping a matrix ##########
########## 转置和重塑矩阵 ##########
print("="*30)
array_41 = np.arange(1,7)
print("array_41: ",array_41)
print("array_41_reshape: \n",array_41.reshape(2,3))
print("array_41_reshape: \n",array_41.reshape(3,2))

array_42 = np.arange(6).reshape((2,3))
print("array_42: ",array_42)
print("array_42_tanspose: ",array_42.transpose())
print("array_42_tanspose: ",array_42.T)


########## How to reverse an array ##########
########## 反转数组 ##########
print("="*30)

### 1 D
array_43 = np.array([1,2,3,4,5,6,7,8])
array_43_reverse = np.flip(array_43)
print("array_43: ",array_43)
print("array_43_reverse: ",array_43_reverse)

### 2 D
array_44 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
array_44_reverse = np.flip(array_44)
print("array_44: ",array_44)
print("array_44_reverse: ",array_44_reverse)
array_44_reverse_rows = np.flip(array_44,axis=0)
print("array_44_reverse_rows: ",array_44_reverse_rows)
array_44_reverse_cols = np.flip(array_44,axis=1)
print("array_44_reverse_cols: ",array_44_reverse_cols)

array_44[1] = np.flip(array_44[1])
print("array_44_reverse[1]: ",array_44)
array_44[:,1] = np.flip(array_44[:,1])
print("array_44_reverse[:,1]: ",array_44)


########## Reshaping and flattening multidimential arrays ##########
########## 重组和展平多维数组 ##########
### flatten
print("="*30)
array_45 = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
array_45_flatten = array_45.flatten()
print("array_45: ",array_45)
print("array_45_flatten: ",array_45_flatten)
array_45_flatten[0] = 99
print("array_45: ",array_45)
print("array_45_flatten: ",array_45_flatten)

### ravel
array_45_ravel = array_45.ravel()
print("array_45: ",array_45)
print("array_45_ravel: ",array_45_ravel)
array_45_ravel[0] = 99
print("array_45: ",array_45)
print("array_45_ravel: ",array_45_ravel)


########## How to access the docstring for more imformation ##########
########## 如何访问文档字符串以获取更多帮助 ##########
print("="*30)
print(help(max))


########## Working with mathematical formulas ##########
########## 使用数学公式 ##########
print("="*30)


########## How to save and load NumPy objects ##########
########## 如何保存和加载 NumPy 对象 ##########
print("="*30)
array_46 = np.array([1, 2, 3, 4, 5, 6])
np.save('filename', array_46)
array_46_load = np.load('filename.npy')
print("array_46_load: ",array_46_load)

array_47 = np.array([1, 2, 3, 4, 5, 6, 7, 8],dtype=np.int32)
np.savetxt('new_file.csv', array_47)
np.savetxt('new_file.txt', array_47)
array_47_load_csv = np.loadtxt('new_file.csv',dtype=np.int32)
array_47_load_txt = np.loadtxt('new_file.txt')
print("array_47_load_csv: ",array_47_load_csv)
print("array_47_load_txt: ",array_47_load_txt)


########## Importing and exporting a CSV ##########
########## 导入和导出 CSV  ##########
print("="*30)
import pandas as pd
array_48 = pd.read_csv('new_file.csv',header=0).values
print("array_48: ",array_48)
# # If all of your columns are the same type:
# x = pd.read_csv('music.csv', header=0).values
# print(x)
# [['Billie Holiday' 'Jazz' 1300000 27000000]
#  ['Jimmie Hendrix' 'Rock' 2700000 70000000]
#  ['Miles Davis' 'Jazz' 1500000 48000000]
#  ['SIA' 'Pop' 2000000 74000000]]
#
# # You can also simply select the columns you need:
# x = pd.read_csv('music.csv', usecols=['Artist', 'Plays']).values
# print(x)
# [['Billie Holiday' 27000000]
#  ['Jimmie Hendrix' 70000000]
#  ['Miles Davis' 48000000]
#  ['SIA' 74000000]]

array_49 = np.array([[-2.58289208,  0.43014843, -1.24082018, 1.59572603],
              [ 0.99027828, 1.17150989,  0.94125714, -0.14692469],
              [ 0.76989341,  0.81299683, -0.95068423, 0.11769564],
              [ 0.20484034,  0.34784527,  1.96979195, 0.51992837]])
array_49_df = pd.DataFrame(array_49)
print("array_49_df: ",array_49_df)
array_49_df.to_csv("pd.csv")
array_49_pd_csv_read = pd.read_csv('pd.csv')
np.savetxt('np.csv', array_49, fmt='%.2f', delimiter=',', header='1,  2,  3,  4')

########## Plotting arrays with Matplotlib ##########
########## 使用 Matplotlib 绘制数组 ##########
print("="*30)
array_50 = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])
import matplotlib.pyplot as plt

###
# plt.plot(array_50)
# plt.show()

###
# x = np.linspace(0, 5, 20)
# y = np.linspace(0, 10, 20)
# plt.plot(x, y, 'purple') # line
# plt.plot(x, y, 'o')      # dots
# plt.show()

###
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X = np.arange(-5, 5, 0.15)
Y = np.arange(-5, 5, 0.15)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
plt.show()

########## linear alegbra 线性代数 ##########
print("="*30)
vec_demo = [3,4,0]
vec_norn = np.linalg.norm(vec_demo)
print("vec_norm: ", vec_norn)
