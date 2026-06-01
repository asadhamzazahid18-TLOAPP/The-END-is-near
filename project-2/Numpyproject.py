import numpy as np

# taking columns from csv file and making them 4 different variables to apply operations on

Avg_Salary, Years_exp, AI_exposure, Tech_factor = np.genfromtxt(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',', unpack=True, usecols=(1, 2, 4, 5), skip_header=1)


print('Average Salary:', Avg_Salary)

print('Years of Experience:', Years_exp)

print('AI Exposure:', AI_exposure)

print('Technology Factor:', Tech_factor)

# Basic operations are being performed on Average Salary column

print('salary mean', np.mean(Avg_Salary))
print('salary  avg', np.average(Avg_Salary))
print('salary std', np.std(Avg_Salary))
print('salary median', np.median(Avg_Salary))
print('salary percentile-25', np.percentile(Avg_Salary, 25))
print('salary percentile-50', np.percentile(Avg_Salary, 50))
print('salary percentile-75', np.percentile(Avg_Salary, 75))
print('salary percentile-3', np.percentile(Avg_Salary, 3))
print(' salary min', np.min(Avg_Salary))
print('salary max', np.max(Avg_Salary))

# Maths operations being performed on Average Salary column

print('salary squareroot', np.sqrt(Avg_Salary))
print('salary square', np.square(Avg_Salary))
print('salary power', np.power(Avg_Salary, 2))
print('salary abs', np.abs(Avg_Salary))

# arithemetic operations on AI_exposure and Tech_factor columns

addition = AI_exposure+Tech_factor
sub = AI_exposure-Tech_factor
div = AI_exposure / Tech_factor
mul = AI_exposure*Tech_factor


print('AI data addition', addition)
print('AI data subtraction', sub)
print('AI data division', div)
print('AI data multiplication', mul)

# Trignometric functions

# The salary was too big giving infinity values so we have compressed it down log1 cause log0 is undefined
# and then divided by pi(3.14159) and +1 so baseline is 1 not 0 and no negative values

Salary_scaled = np.log1p(Avg_Salary)
Salary_pie = (Salary_scaled / np.pi) + 1

print('Ai dataset sin', np.sin(Salary_pie))
print('Ai dataset cos', np.cos(Salary_pie))
print('Ai dataset tan', np.tan(Salary_pie))

print('Ai dataset exp', np.exp(Salary_pie))

# Logrithm
print('AI dataset Log', np.log(Salary_pie))
print('AI dataset Log10', np.log10(Salary_pie))

# Calculate the hyperbolic sine , cosh and tangent values of the salarypie variable we made B4

print('AI sinh', np.sinh(Salary_pie))

print('AI cosh', np.cosh(Salary_pie))

print('AI tangent', np.tanh(Salary_pie))

# now inverse of hyperbolic values on salarypie variable

print('AI dataset arc sin', np.arcsinh(Salary_pie))

print('AI dataset arc cos', np.arccosh(Salary_pie))

# array creating of two columns AI_exposure and Tech_factor

D2AI = np.array([AI_exposure, Tech_factor])

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

# there were too many elements and looping keeps on going so we have taken out only 5 columns values to keep the looping and not make it infinte at the same time

D2AI1 = D2AI[:, :5]

for elem in np.nditer(D2AI1):
    print(elem)


# if you need indexes for tuple or 2d table

for index, elem in np.ndenumerate(D2AI1):
    print(index, elem)

# before shape was (2,3000) after reshape into 1d (1,6000) cause we have 2 rows and 3000 columns so total 6000 values in the array

D2AIshape = np.reshape(D2AI, (1, 6000))

print('1D shape is', D2AIshape)
print('shape is', D2AIshape.shape)
print('size is', D2AIshape.size)
print('dimension is', D2AIshape.ndim)

print('----------')
print('Numpy ends here!!!')
