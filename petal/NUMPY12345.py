import numpy as np

# loading the data from csv file and unpacking it into separate arrays for each column, filling missing values with NaN and skipping the header row and selecting specific columns (2, 3, 4, 6) for analysis
RD_Spending_USD_Mn, AI_Revenue_USD_Mn, AI_Revenue_Growth, Stock_Impact = np.genfromtxt(
    'petal/ai_financial_market_daily_realistic_synthetic.csv', delimiter=',', unpack=True, filling_values=np.nan, usecols=(2, 3, 4, 6), skip_header=1)

# printing the loaded data
print(RD_Spending_USD_Mn)

print(AI_Revenue_USD_Mn)

print(AI_Revenue_Growth)

print(Stock_Impact)

# checking the type of the loaded data
print(type(RD_Spending_USD_Mn))

# Basic operations are being performed

print('Research department investment mean', np.mean(RD_Spending_USD_Mn))
print('Research department investment average', np.average(RD_Spending_USD_Mn))
print('Research department investment std', np.std(RD_Spending_USD_Mn))
print('Research department investment median', np.median(RD_Spending_USD_Mn))
print('Research department investment percentile-25',
      np.percentile(RD_Spending_USD_Mn, 25))
print('Research department investment percentile-50',
      np.percentile(RD_Spending_USD_Mn, 50))
print('Research department investment percentile-75',
      np.percentile(RD_Spending_USD_Mn, 75))
print('Research department investment percentile-3',
      np.percentile(RD_Spending_USD_Mn, 3))
print('Research department investment min', np.min(RD_Spending_USD_Mn))
print('Research department investment max', np.max(RD_Spending_USD_Mn))

# Maths operations

print('Research department investment squareroot', np.sqrt(RD_Spending_USD_Mn))
print('Research department investment square', np.square(RD_Spending_USD_Mn))
print('Research department investment power', np.power(RD_Spending_USD_Mn, 2))
print('Research department investment abs', np.abs(RD_Spending_USD_Mn))

# arithemetic operations

addition = AI_Revenue_USD_Mn + AI_Revenue_Growth
sub = AI_Revenue_USD_Mn + AI_Revenue_Growth

# as there were zeros causing error so we have applied this step so no row will be skipped and zero values will be replaced
# AI_ROI_safe = np.where(AI_ROI_percen == 0, 1e-8, AI_ROI_percen)

div = AI_Revenue_USD_Mn + AI_Revenue_Growth
mul = AI_Revenue_USD_Mn + AI_Revenue_Growth


print('AI stocks addition', addition)
print('AI stocks subraction', sub)
print('AI stocks division', div)
print('AI stocks multiplication', mul)

# Trignometric functions

# The Revenue was too big giving infinity values so we have compressed it down log1 cause log0 is undefined
# and the divided by pi and +1 so baseline is 1 not 0 and no negative values

RD_Spending_USD_Mn = np.where(
    RD_Spending_USD_Mn <= 0, np.nan, RD_Spending_USD_Mn)

RD_Spending_scaled = np.log(RD_Spending_USD_Mn)

RDpie = (RD_Spending_scaled / np.pi) + 1


print('Research developement/ dataset sin', np.sin(RDpie))
print('Research developement dataset cos', np.cos(RDpie))
print('Research developement dataset tan', np.tan(RDpie))

print('Research developement dataset exp', np.exp(RDpie))

# Logrithm
safe_input = np.where(RDpie >= 1, RDpie, 1)

print('Research developement dataset Log', np.log(safe_input))
print('Research developement dataset Log10', np.log10(safe_input))

# Calculate the hyperbolic sine , cosh and tangent

print('Research developement sinh', np.sinh(RDpie))

print('Research developement cosh', np.cosh(RDpie))

print('Research developement tangent', np.tanh(RDpie))

# now inverse of hyperbolic values

# safe_input = np.where(closepie >= 1, closepie, 1)

print('AI dataset arc cos', np.arccosh(safe_input))

print('AI dataset arc sin', np.arcsinh(safe_input))

print('AI dataset arc cos', np.arccosh(safe_input))

# array


D2AI = np.array([AI_Revenue_Growth, AI_Revenue_USD_Mn])

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

D2AIshape = np.reshape(D2AI, (1, 21918))

print('1D shape is', D2AIshape)
print('shape is', D2AIshape.shape)
print('size is', D2AIshape.size)
print('dimension is', D2AIshape.ndim)

print('----------')
