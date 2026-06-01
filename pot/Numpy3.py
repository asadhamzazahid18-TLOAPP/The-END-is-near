import numpy as np

# loading the data from csv file and unpacking it into different variables
high, open, low, close, volume, adjusted = np.genfromtxt(
    'pot/top_10_ai_stocks.csv', delimiter=',', unpack=True, usecols=(2, 3, 4, 5, 7, 6), skip_header=1)


print(close)

# as there will be missing values so,
# we have replace nan with 1 of column close
close = np.nan_to_num(close, nan=1)


print('High values:', high)

print('Low values:', low)

print('Volume values:', volume)

print('Adjusted values:', adjusted)

print('Close values type:', type(close))


# Basic operations are being performed

print('close mean', np.mean(close))
print('close avg', np.average(close))
print('close std', np.std(close))
print('close median', np.median(close))
print('close percentile-25', np.percentile(close, 25))
print('close percentile-50', np.percentile(close, 50))
print('close percentile-75', np.percentile(close, 75))
print('close percentile-3', np.percentile(close, 3))
print('close min', np.min(close))
print('close max', np.max(close))

# Maths operations

print('close squareroot', np.sqrt(close))
print('close square', np.square(close))
print('close power', np.power(close, 2))
print('close abs', np.abs(close))

# arithemetic operations

addition = high+low
sub = high-low

# as there were zeros causing error so we have applied this step so no row will be skipped and zero values will be replaced
# AI_ROI_safe = np.where(AI_ROI_percen == 0, 1e-8, AI_ROI_percen)

div = (high-low)/close
mul = close*volume


print('AI stocks addition', addition)
print('AI stocks subraction', sub)
print('AI stocks division', div)
print('AI stocks multiplication', mul)

# Trignometric functions

# The Revenue was too big giving infinity values so we have compressed it down log1 cause log0 is undefined
# and the divided by pi and +1 so baseline is 1 not 0 and no negative values

close = np.where(close <= 0, np.nan, close)

close_scaled = np.log(close)

closepie = (close_scaled / np.pi) + 1


print('Ai dataset sin', np.sin(closepie))
print('Ai dataset cos', np.cos(closepie))
print('Ai dataset tan', np.tan(closepie))

print('Ai dataset exp', np.exp(closepie))

# Logrithm
safe_input = np.where(closepie >= 1, closepie, 1)

print('AI dataset Log', np.log(safe_input))
print('AI dataset Log10', np.log10(safe_input))

# Calculate the hyperbolic sine , cosh and tangent

print('AI sinh', np.sinh(closepie))

print('AI cosh', np.cosh(closepie))

print('AI tangent', np.tanh(closepie))

# now inverse of hyperbolic values

# safe_input = np.where(closepie >= 1, closepie, 1)

print('AI dataset arc cos', np.arccosh(safe_input))

print('AI dataset arc sin', np.arcsinh(safe_input))

print('AI dataset arc cos', np.arccosh(safe_input))

# array


D2AI = np.array([open, close])

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

D2AIshape = np.reshape(D2AI, (1, 77472))

print('1D shape is', D2AIshape)
print('shape is', D2AIshape.shape)
print('size is', D2AIshape.size)
print('dimension is', D2AIshape.ndim)

print('----------')
