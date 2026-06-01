import numpy as np


# loading the dataset using genfromtxt function and unpacking the columns into separate variables
year, Revenue, AI_ROI_percen, AI_matur_sco = np.genfromtxt(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',', unpack=True, usecols=(0, 6, 9, 10), skip_header=1)

print(year)

print(Revenue)

print(AI_ROI_percen)

print(AI_matur_sco)

# Basic operations are being performed

print('AI dataset mean', np.mean(Revenue))
print('AI dataset avg', np.average(Revenue))
print('AI dataset std', np.std(Revenue))
print('AI dataset median', np.median(Revenue))
print('AI dataset percentile-25', np.percentile(Revenue, 25))
print('AI dataset percentile-50', np.percentile(Revenue, 50))
print('AI dataset percentile-75', np.percentile(Revenue, 75))
print('AI dataset percentile-3', np.percentile(Revenue, 3))
print('AI dataset min', np.min(Revenue))
print('AI dataset max', np.max(Revenue))

# Maths operations

print('AI dataset squareroot', np.sqrt(Revenue))
print('AI dataset square', np.square(Revenue))
print('AI dataset power', np.power(Revenue, 2))
print('AI dataset abs', np.abs(Revenue))

# arithemetic operations

addition = AI_matur_sco+AI_ROI_percen
sub = AI_matur_sco-AI_ROI_percen

# as there were zeros causing error so we have applied this step so no row will be skipped and zero values will be replaced
AI_ROI_safe = np.where(AI_ROI_percen == 0, 1e-8, AI_ROI_percen)
div = AI_matur_sco / AI_ROI_safe
mul = AI_matur_sco*AI_ROI_percen


print('AI data addition', addition)
print('AI data subraction', sub)
print('AI data division', div)
print('AI data multiplication', mul)

# Trignometric functions

# The Revenue was too big giving infinity values so we have compressed it down log1 cause log0 is undefined
# and the divided by pi and +1 so baseline is 1 not 0 and no negative values

Revenue_scaled = np.log1p(Revenue)
Revenuepie = (Revenue_scaled / np.pi) + 1

print('Ai dataset sin', np.sin(Revenuepie))
print('Ai dataset cos', np.cos(Revenuepie))
print('Ai dataset tan', np.tan(Revenuepie))

print('Ai dataset exp', np.exp(Revenuepie))

# Logrithm

print('AI dataset Log', np.log(Revenuepie))
print('AI dataset Log10', np.log10(Revenuepie))

# Calculate the hyperbolic sine , cosh and tangent

print('AI sinh', np.sinh(Revenuepie))

print('AI cosh', np.cosh(Revenuepie))

print('AI tangent', np.tanh(Revenuepie))

# now inverse of hyperbolic values

print('AI dataset arc sin', np.arcsinh(Revenuepie))

print('AI dataset arc cos', np.arccosh(Revenuepie))

# array


D2AI = np.array([AI_matur_sco, AI_ROI_percen])

print('2 dimensional array is', D2AI)

# dimensions

print('Array dimenstions is', D2AI.ndim)

# sizeofarray

print('Array size is', D2AI.size)

# shapeofarrayis

print('Array shape is', D2AI.shape)

# datatypeofarray

print('Array datatype is', D2AI.dtype)

# SLICING

slice = D2AI[:1, :8]
print('Slicing it ', slice)

slice2 = D2AI[:1, 1:10:]
print(slice2)

# indexing

sliceitemonly = D2AI[1, 2]
sliceitemonly3 = D2AI[0, 10]

print('indexing of this array is', sliceitemonly)
print('indexing of this array1 is', sliceitemonly3)

# using nditer function cause it will got through each value sequentianlly

# there were too many elements and looping keeps on going so we have taken out only 10 values to keep the looping and not make it infinte at the same time

D2AI1 = D2AI[:, :10]

for elem in np.nditer(D2AI1):
    print(elem)


# if you need indexes for tuple or 2d table

for index, elem in np.ndenumerate(D2AI1):
    print(index, elem)

# before shape was (2,6000) after reshape into 1d (1,12000)

D2AIshape = np.reshape(D2AI, (1, 12000))

print('1D shape is', D2AIshape)
print('shape is', D2AIshape.shape)
print('size is', D2AIshape.size)
print('dimension is', D2AIshape.ndim)

print('----------')
