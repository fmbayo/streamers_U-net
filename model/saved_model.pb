эћ4
п
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
М
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628и-
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0

Adam/v/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_9/bias
y
(Adam/v/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_9/bias
y
(Adam/m/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_9/kernel

*Adam/v/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_9/kernel

*Adam/m/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/kernel*&
_output_shapes
:*
dtype0

!Adam/v/group_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/group_normalization_8/beta

5Adam/v/group_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_8/beta*
_output_shapes
:*
dtype0

!Adam/m/group_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/group_normalization_8/beta

5Adam/m/group_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_8/beta*
_output_shapes
:*
dtype0

"Adam/v/group_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/group_normalization_8/gamma

6Adam/v/group_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_8/gamma*
_output_shapes
:*
dtype0

"Adam/m/group_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/group_normalization_8/gamma

6Adam/m/group_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_8/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_8/bias
y
(Adam/v/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_8/bias
y
(Adam/m/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_8/kernel

*Adam/v/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_8/kernel

*Adam/m/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/kernel*&
_output_shapes
:*
dtype0

!Adam/v/group_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/group_normalization_7/beta

5Adam/v/group_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_7/beta*
_output_shapes
:*
dtype0

!Adam/m/group_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/group_normalization_7/beta

5Adam/m/group_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_7/beta*
_output_shapes
:*
dtype0

"Adam/v/group_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/group_normalization_7/gamma

6Adam/v/group_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_7/gamma*
_output_shapes
:*
dtype0

"Adam/m/group_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/group_normalization_7/gamma

6Adam/m/group_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_7/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_7/bias
y
(Adam/v/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_7/bias
y
(Adam/m/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_7/kernel

*Adam/v/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/kernel*&
_output_shapes
: *
dtype0

Adam/m/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_7/kernel

*Adam/m/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/kernel*&
_output_shapes
: *
dtype0

!Adam/v/group_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/group_normalization_6/beta

5Adam/v/group_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_6/beta*
_output_shapes
:*
dtype0

!Adam/m/group_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/group_normalization_6/beta

5Adam/m/group_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_6/beta*
_output_shapes
:*
dtype0

"Adam/v/group_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/group_normalization_6/gamma

6Adam/v/group_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_6/gamma*
_output_shapes
:*
dtype0

"Adam/m/group_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/group_normalization_6/gamma

6Adam/m/group_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_6/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_6/bias
y
(Adam/v/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_6/bias
y
(Adam/m/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_6/kernel

*Adam/v/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/kernel*&
_output_shapes
:@*
dtype0

Adam/m/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_6/kernel

*Adam/m/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/kernel*&
_output_shapes
:@*
dtype0

!Adam/v/group_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/group_normalization_5/beta

5Adam/v/group_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_5/beta*
_output_shapes
: *
dtype0

!Adam/m/group_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/group_normalization_5/beta

5Adam/m/group_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_5/beta*
_output_shapes
: *
dtype0

"Adam/v/group_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/group_normalization_5/gamma

6Adam/v/group_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_5/gamma*
_output_shapes
: *
dtype0

"Adam/m/group_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/group_normalization_5/gamma

6Adam/m/group_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_5/gamma*
_output_shapes
: *
dtype0

Adam/v/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_5/bias
y
(Adam/v/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/bias*
_output_shapes
: *
dtype0

Adam/m/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_5/bias
y
(Adam/m/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/bias*
_output_shapes
: *
dtype0

Adam/v/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/v/conv2d_5/kernel

*Adam/v/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/kernel*&
_output_shapes
:@ *
dtype0

Adam/m/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/m/conv2d_5/kernel

*Adam/m/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/kernel*&
_output_shapes
:@ *
dtype0

!Adam/v/group_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/group_normalization_4/beta

5Adam/v/group_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_4/beta*
_output_shapes
:@*
dtype0

!Adam/m/group_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/group_normalization_4/beta

5Adam/m/group_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_4/beta*
_output_shapes
:@*
dtype0

"Adam/v/group_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/group_normalization_4/gamma

6Adam/v/group_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_4/gamma*
_output_shapes
:@*
dtype0

"Adam/m/group_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/group_normalization_4/gamma

6Adam/m/group_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_4/gamma*
_output_shapes
:@*
dtype0

Adam/v/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_4/bias
y
(Adam/v/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/bias*
_output_shapes
:@*
dtype0

Adam/m/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_4/bias
y
(Adam/m/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/bias*
_output_shapes
:@*
dtype0

Adam/v/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/v/conv2d_4/kernel

*Adam/v/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/kernel*&
_output_shapes
: @*
dtype0

Adam/m/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/m/conv2d_4/kernel

*Adam/m/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/kernel*&
_output_shapes
: @*
dtype0

!Adam/v/group_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/group_normalization_3/beta

5Adam/v/group_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_3/beta*
_output_shapes
: *
dtype0

!Adam/m/group_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/group_normalization_3/beta

5Adam/m/group_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_3/beta*
_output_shapes
: *
dtype0

"Adam/v/group_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/group_normalization_3/gamma

6Adam/v/group_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_3/gamma*
_output_shapes
: *
dtype0

"Adam/m/group_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/group_normalization_3/gamma

6Adam/m/group_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_3/gamma*
_output_shapes
: *
dtype0

Adam/v/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_3/bias
y
(Adam/v/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/bias*
_output_shapes
: *
dtype0

Adam/m/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_3/bias
y
(Adam/m/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/bias*
_output_shapes
: *
dtype0

Adam/v/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_3/kernel

*Adam/v/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/kernel*&
_output_shapes
: *
dtype0

Adam/m/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_3/kernel

*Adam/m/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/kernel*&
_output_shapes
: *
dtype0

!Adam/v/group_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/group_normalization_2/beta

5Adam/v/group_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_2/beta*
_output_shapes
:*
dtype0

!Adam/m/group_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/group_normalization_2/beta

5Adam/m/group_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_2/beta*
_output_shapes
:*
dtype0

"Adam/v/group_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/group_normalization_2/gamma

6Adam/v/group_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_2/gamma*
_output_shapes
:*
dtype0

"Adam/m/group_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/group_normalization_2/gamma

6Adam/m/group_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_2/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_2/bias
y
(Adam/v/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_2/bias
y
(Adam/m/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_2/kernel

*Adam/v/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_2/kernel

*Adam/m/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/kernel*&
_output_shapes
:*
dtype0

!Adam/v/group_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/group_normalization_1/beta

5Adam/v/group_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/group_normalization_1/beta*
_output_shapes
:*
dtype0

!Adam/m/group_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/group_normalization_1/beta

5Adam/m/group_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/group_normalization_1/beta*
_output_shapes
:*
dtype0

"Adam/v/group_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/group_normalization_1/gamma

6Adam/v/group_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/group_normalization_1/gamma*
_output_shapes
:*
dtype0

"Adam/m/group_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/group_normalization_1/gamma

6Adam/m/group_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/group_normalization_1/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_1/bias
y
(Adam/v/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_1/bias
y
(Adam/m/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_1/kernel

*Adam/v/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_1/kernel

*Adam/m/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/kernel*&
_output_shapes
:*
dtype0

Adam/v/group_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/group_normalization/beta

3Adam/v/group_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/group_normalization/beta*
_output_shapes
:*
dtype0

Adam/m/group_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/group_normalization/beta

3Adam/m/group_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/group_normalization/beta*
_output_shapes
:*
dtype0

 Adam/v/group_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/group_normalization/gamma

4Adam/v/group_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/group_normalization/gamma*
_output_shapes
:*
dtype0

 Adam/m/group_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/group_normalization/gamma

4Adam/m/group_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/group_normalization/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv2d/bias
u
&Adam/v/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv2d/bias
u
&Adam/m/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d/kernel

(Adam/v/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d/kernel

(Adam/m/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0

group_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namegroup_normalization_8/beta

.group_normalization_8/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_8/beta*
_output_shapes
:*
dtype0

group_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namegroup_normalization_8/gamma

/group_normalization_8/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_8/gamma*
_output_shapes
:*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0

group_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namegroup_normalization_7/beta

.group_normalization_7/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_7/beta*
_output_shapes
:*
dtype0

group_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namegroup_normalization_7/gamma

/group_normalization_7/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_7/gamma*
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: *
dtype0

group_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namegroup_normalization_6/beta

.group_normalization_6/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_6/beta*
_output_shapes
:*
dtype0

group_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namegroup_normalization_6/gamma

/group_normalization_6/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_6/gamma*
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@*
dtype0

group_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namegroup_normalization_5/beta

.group_normalization_5/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_5/beta*
_output_shapes
: *
dtype0

group_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namegroup_normalization_5/gamma

/group_normalization_5/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_5/gamma*
_output_shapes
: *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@ *
dtype0

group_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namegroup_normalization_4/beta

.group_normalization_4/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_4/beta*
_output_shapes
:@*
dtype0

group_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namegroup_normalization_4/gamma

/group_normalization_4/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_4/gamma*
_output_shapes
:@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: @*
dtype0

group_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namegroup_normalization_3/beta

.group_normalization_3/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_3/beta*
_output_shapes
: *
dtype0

group_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namegroup_normalization_3/gamma

/group_normalization_3/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_3/gamma*
_output_shapes
: *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0

group_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namegroup_normalization_2/beta

.group_normalization_2/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_2/beta*
_output_shapes
:*
dtype0

group_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namegroup_normalization_2/gamma

/group_normalization_2/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_2/gamma*
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0

group_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namegroup_normalization_1/beta

.group_normalization_1/beta/Read/ReadVariableOpReadVariableOpgroup_normalization_1/beta*
_output_shapes
:*
dtype0

group_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namegroup_normalization_1/gamma

/group_normalization_1/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization_1/gamma*
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0

group_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namegroup_normalization/beta

,group_normalization/beta/Read/ReadVariableOpReadVariableOpgroup_normalization/beta*
_output_shapes
:*
dtype0

group_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namegroup_normalization/gamma

-group_normalization/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization/gamma*
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
Ў
serving_default_input_1Placeholder*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*6
shape-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
я	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasgroup_normalization/gammagroup_normalization/betaconv2d_1/kernelconv2d_1/biasgroup_normalization_1/gammagroup_normalization_1/betaconv2d_2/kernelconv2d_2/biasgroup_normalization_2/gammagroup_normalization_2/betaconv2d_3/kernelconv2d_3/biasgroup_normalization_3/gammagroup_normalization_3/betaconv2d_4/kernelconv2d_4/biasgroup_normalization_4/gammagroup_normalization_4/betaconv2d_5/kernelconv2d_5/biasgroup_normalization_5/gammagroup_normalization_5/betaconv2d_6/kernelconv2d_6/biasgroup_normalization_6/gammagroup_normalization_6/betaconv2d_7/kernelconv2d_7/biasgroup_normalization_7/gammagroup_normalization_7/betaconv2d_8/kernelconv2d_8/biasgroup_normalization_8/gammagroup_normalization_8/betaconv2d_9/kernelconv2d_9/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_6467382

NoOpNoOp
Л
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*жК
valueЫКBЧК BПК


layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer-25
layer-26
layer_with_weights-12
layer-27
layer_with_weights-13
layer-28
layer-29
layer-30
 layer-31
!layer_with_weights-14
!layer-32
"layer_with_weights-15
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-16
&layer-37
'layer_with_weights-17
'layer-38
(layer-39
)layer-40
*layer-41
+layer_with_weights-18
+layer-42
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_default_save_signature
3	optimizer
4
signatures*
* 
Ш
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*

>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
Ѕ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
	Jgamma
Kbeta*

L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
Ш
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op*

[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
Ѕ
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
	ggamma
hbeta*

i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
Ш
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias
 w_jit_compiled_convolution_op*

x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 
Ћ
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

gamma
	beta*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
­
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

Ёgamma
	Ђbeta*

Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses* 
б
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
Џkernel
	Аbias
!Б_jit_compiled_convolution_op*

В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses* 
­
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses

Оgamma
	Пbeta*

Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses* 

Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses* 
б
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
вkernel
	гbias
!д_jit_compiled_convolution_op*
­
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses

лgamma
	мbeta*

н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses* 

у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses* 

щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses* 
б
я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses
ѕkernel
	іbias
!ї_jit_compiled_convolution_op*
­
ј	variables
љtrainable_variables
њregularization_losses
ћ	keras_api
ќ__call__
+§&call_and_return_all_conditional_losses

ўgamma
	џbeta*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
­
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

Ёgamma
	Ђbeta*

Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses* 

Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses* 

Џ	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses* 
б
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
Лkernel
	Мbias
!Н_jit_compiled_convolution_op*
­
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses

Фgamma
	Хbeta*

Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses* 

Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses* 

в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses* 
б
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses
оkernel
	пbias
!р_jit_compiled_convolution_op*
Ц
;0
<1
J2
K3
X4
Y5
g6
h7
u8
v9
10
11
12
13
Ё14
Ђ15
Џ16
А17
О18
П19
в20
г21
л22
м23
ѕ24
і25
ў26
џ27
28
29
Ё30
Ђ31
Л32
М33
Ф34
Х35
о36
п37*
Ц
;0
<1
J2
K3
X4
Y5
g6
h7
u8
v9
10
11
12
13
Ё14
Ђ15
Џ16
А17
О18
П19
в20
г21
л22
м23
ѕ24
і25
ў26
џ27
28
29
Ё30
Ђ31
Л32
М33
Ф34
Х35
о36
п37*
* 
Е
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
2_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

цtrace_0
чtrace_1* 

шtrace_0
щtrace_1* 
* 

ъ
_variables
ы_iterations
ь_learning_rate
э_index_dict
ю
_momentums
я_velocities
№_update_step_xla*

ёserving_default* 

;0
<1*

;0
<1*
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

їtrace_0* 

јtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

ўtrace_0* 

џtrace_0* 

J0
K1*

J0
K1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

trace_0* 

trace_0* 
hb
VARIABLE_VALUEgroup_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEgroup_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

X0
Y1*

X0
Y1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

g0
h1*

g0
h1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

Ёtrace_0* 

Ђtrace_0* 
jd
VARIABLE_VALUEgroup_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEgroup_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ѓnon_trainable_variables
Єlayers
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

Јtrace_0* 

Љtrace_0* 

u0
v1*

u0
v1*
* 

Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

Џtrace_0* 

Аtrace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 

0
1*

0
1*
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Нtrace_0* 

Оtrace_0* 
jd
VARIABLE_VALUEgroup_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEgroup_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Фtrace_0* 

Хtrace_0* 

0
1*

0
1*
* 

Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

вtrace_0* 

гtrace_0* 

Ё0
Ђ1*

Ё0
Ђ1*
* 

дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

йtrace_0* 

кtrace_0* 
jd
VARIABLE_VALUEgroup_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEgroup_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses* 

рtrace_0* 

сtrace_0* 

Џ0
А1*

Џ0
А1*
* 

тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*

чtrace_0* 

шtrace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses* 

юtrace_0* 

яtrace_0* 

О0
П1*

О0
П1*
* 

№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses*

ѕtrace_0* 

іtrace_0* 
jd
VARIABLE_VALUEgroup_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEgroup_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

їnon_trainable_variables
јlayers
љmetrics
 њlayer_regularization_losses
ћlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses* 

ќtrace_0* 

§trace_0* 
* 
* 
* 

ўnon_trainable_variables
џlayers
metrics
 layer_regularization_losses
layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

в0
г1*

в0
г1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

л0
м1*

л0
м1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses*

trace_0* 

trace_0* 
ke
VARIABLE_VALUEgroup_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEgroup_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
у	variables
фtrainable_variables
хregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses* 

trace_0* 

 trace_0* 
* 
* 
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses* 

Іtrace_0* 

Їtrace_0* 

ѕ0
і1*

ѕ0
і1*
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses*

­trace_0* 

Ўtrace_0* 
`Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

ў0
џ1*

ў0
џ1*
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
ј	variables
љtrainable_variables
њregularization_losses
ќ__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
ke
VARIABLE_VALUEgroup_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEgroup_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Лtrace_0* 

Мtrace_0* 
* 
* 
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Тtrace_0* 

Уtrace_0* 
* 
* 
* 

Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Щtrace_0* 

Ъtrace_0* 

0
1*

0
1*
* 

Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

аtrace_0* 

бtrace_0* 
`Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ё0
Ђ1*

Ё0
Ђ1*
* 

вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

зtrace_0* 

иtrace_0* 
ke
VARIABLE_VALUEgroup_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEgroup_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses* 

оtrace_0* 

пtrace_0* 
* 
* 
* 

рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses* 

хtrace_0* 

цtrace_0* 
* 
* 
* 

чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
Џ	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses* 

ьtrace_0* 

эtrace_0* 

Л0
М1*

Л0
М1*
* 

юnon_trainable_variables
яlayers
№metrics
 ёlayer_regularization_losses
ђlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*

ѓtrace_0* 

єtrace_0* 
`Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ф0
Х1*

Ф0
Х1*
* 

ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*

њtrace_0* 

ћtrace_0* 
ke
VARIABLE_VALUEgroup_normalization_8/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEgroup_normalization_8/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ќnon_trainable_variables
§layers
ўmetrics
 џlayer_regularization_losses
layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

о0
п1*

о0
п1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_9/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_9/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
в
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42*

0*
* 
* 
* 
* 
* 
* 
Џ
ы0
1
2
3
4
5
6
7
 8
Ё9
Ђ10
Ѓ11
Є12
Ѕ13
І14
Ї15
Ј16
Љ17
Њ18
Ћ19
Ќ20
­21
Ў22
Џ23
А24
Б25
В26
Г27
Д28
Е29
Ж30
З31
И32
Й33
К34
Л35
М36
Н37
О38
П39
Р40
С41
Т42
У43
Ф44
Х45
Ц46
Ч47
Ш48
Щ49
Ъ50
Ы51
Ь52
Э53
Ю54
Я55
а56
б57
в58
г59
д60
е61
ж62
з63
и64
й65
к66
л67
м68
н69
о70
п71
р72
с73
т74
у75
ф76*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
а
0
1
2
3
Ё4
Ѓ5
Ѕ6
Ї7
Љ8
Ћ9
­10
Џ11
Б12
Г13
Е14
З15
Й16
Л17
Н18
П19
С20
У21
Х22
Ч23
Щ24
Ы25
Э26
Я27
б28
г29
е30
з31
й32
л33
н34
п35
с36
у37*
а
0
1
2
 3
Ђ4
Є5
І6
Ј7
Њ8
Ќ9
Ў10
А11
В12
Д13
Ж14
И15
К16
М17
О18
Р19
Т20
Ф21
Ц22
Ш23
Ъ24
Ь25
Ю26
а27
в28
д29
ж30
и31
к32
м33
о34
р35
т36
ф37*
В
хtrace_0
цtrace_1
чtrace_2
шtrace_3
щtrace_4
ъtrace_5
ыtrace_6
ьtrace_7
эtrace_8
юtrace_9
яtrace_10
№trace_11
ёtrace_12
ђtrace_13
ѓtrace_14
єtrace_15
ѕtrace_16
іtrace_17
їtrace_18
јtrace_19
љtrace_20
њtrace_21
ћtrace_22
ќtrace_23
§trace_24
ўtrace_25
џtrace_26
trace_27
trace_28
trace_29
trace_30
trace_31
trace_32
trace_33
trace_34
trace_35
trace_36
trace_37* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
_Y
VARIABLE_VALUEAdam/m/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/group_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/group_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/group_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/group_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_4/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_4/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_4/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_4/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_4/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_4/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_4/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_4/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_5/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_5/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_5/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_5/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_5/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_5/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_5/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_5/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_6/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_6/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_6/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_6/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_6/gamma2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_6/gamma2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_6/beta2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_6/beta2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_7/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_7/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_7/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_7/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_7/gamma2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_7/gamma2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_7/beta2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_7/beta2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_8/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_8/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_8/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_8/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/group_normalization_8/gamma2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/group_normalization_8/gamma2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/group_normalization_8/beta2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/group_normalization_8/beta2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_9/kernel2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_9/kernel2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_9/bias2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_9/bias2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasgroup_normalization/gammagroup_normalization/betaconv2d_1/kernelconv2d_1/biasgroup_normalization_1/gammagroup_normalization_1/betaconv2d_2/kernelconv2d_2/biasgroup_normalization_2/gammagroup_normalization_2/betaconv2d_3/kernelconv2d_3/biasgroup_normalization_3/gammagroup_normalization_3/betaconv2d_4/kernelconv2d_4/biasgroup_normalization_4/gammagroup_normalization_4/betaconv2d_5/kernelconv2d_5/biasgroup_normalization_5/gammagroup_normalization_5/betaconv2d_6/kernelconv2d_6/biasgroup_normalization_6/gammagroup_normalization_6/betaconv2d_7/kernelconv2d_7/biasgroup_normalization_7/gammagroup_normalization_7/betaconv2d_8/kernelconv2d_8/biasgroup_normalization_8/gammagroup_normalization_8/betaconv2d_9/kernelconv2d_9/bias	iterationlearning_rateAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias Adam/m/group_normalization/gamma Adam/v/group_normalization/gammaAdam/m/group_normalization/betaAdam/v/group_normalization/betaAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/group_normalization_1/gamma"Adam/v/group_normalization_1/gamma!Adam/m/group_normalization_1/beta!Adam/v/group_normalization_1/betaAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/bias"Adam/m/group_normalization_2/gamma"Adam/v/group_normalization_2/gamma!Adam/m/group_normalization_2/beta!Adam/v/group_normalization_2/betaAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/bias"Adam/m/group_normalization_3/gamma"Adam/v/group_normalization_3/gamma!Adam/m/group_normalization_3/beta!Adam/v/group_normalization_3/betaAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias"Adam/m/group_normalization_4/gamma"Adam/v/group_normalization_4/gamma!Adam/m/group_normalization_4/beta!Adam/v/group_normalization_4/betaAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/bias"Adam/m/group_normalization_5/gamma"Adam/v/group_normalization_5/gamma!Adam/m/group_normalization_5/beta!Adam/v/group_normalization_5/betaAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias"Adam/m/group_normalization_6/gamma"Adam/v/group_normalization_6/gamma!Adam/m/group_normalization_6/beta!Adam/v/group_normalization_6/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias"Adam/m/group_normalization_7/gamma"Adam/v/group_normalization_7/gamma!Adam/m/group_normalization_7/beta!Adam/v/group_normalization_7/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias"Adam/m/group_normalization_8/gamma"Adam/v/group_normalization_8/gamma!Adam/m/group_normalization_8/beta!Adam/v/group_normalization_8/betaAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/biastotalcountConst*
Tin|
z2x*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *)
f$R"
 __inference__traced_save_6469272

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasgroup_normalization/gammagroup_normalization/betaconv2d_1/kernelconv2d_1/biasgroup_normalization_1/gammagroup_normalization_1/betaconv2d_2/kernelconv2d_2/biasgroup_normalization_2/gammagroup_normalization_2/betaconv2d_3/kernelconv2d_3/biasgroup_normalization_3/gammagroup_normalization_3/betaconv2d_4/kernelconv2d_4/biasgroup_normalization_4/gammagroup_normalization_4/betaconv2d_5/kernelconv2d_5/biasgroup_normalization_5/gammagroup_normalization_5/betaconv2d_6/kernelconv2d_6/biasgroup_normalization_6/gammagroup_normalization_6/betaconv2d_7/kernelconv2d_7/biasgroup_normalization_7/gammagroup_normalization_7/betaconv2d_8/kernelconv2d_8/biasgroup_normalization_8/gammagroup_normalization_8/betaconv2d_9/kernelconv2d_9/bias	iterationlearning_rateAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias Adam/m/group_normalization/gamma Adam/v/group_normalization/gammaAdam/m/group_normalization/betaAdam/v/group_normalization/betaAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/group_normalization_1/gamma"Adam/v/group_normalization_1/gamma!Adam/m/group_normalization_1/beta!Adam/v/group_normalization_1/betaAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/bias"Adam/m/group_normalization_2/gamma"Adam/v/group_normalization_2/gamma!Adam/m/group_normalization_2/beta!Adam/v/group_normalization_2/betaAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/bias"Adam/m/group_normalization_3/gamma"Adam/v/group_normalization_3/gamma!Adam/m/group_normalization_3/beta!Adam/v/group_normalization_3/betaAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias"Adam/m/group_normalization_4/gamma"Adam/v/group_normalization_4/gamma!Adam/m/group_normalization_4/beta!Adam/v/group_normalization_4/betaAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/bias"Adam/m/group_normalization_5/gamma"Adam/v/group_normalization_5/gamma!Adam/m/group_normalization_5/beta!Adam/v/group_normalization_5/betaAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias"Adam/m/group_normalization_6/gamma"Adam/v/group_normalization_6/gamma!Adam/m/group_normalization_6/beta!Adam/v/group_normalization_6/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias"Adam/m/group_normalization_7/gamma"Adam/v/group_normalization_7/gamma!Adam/m/group_normalization_7/beta!Adam/v/group_normalization_7/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias"Adam/m/group_normalization_8/gamma"Adam/v/group_normalization_8/gamma!Adam/m/group_normalization_8/beta!Adam/v/group_normalization_8/betaAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/biastotalcount*
Tin{
y2w*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *,
f'R%
#__inference__traced_restore_6469635ѓ(
Ђ
h
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6466461

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44436
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_44396
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Г
ў
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6466848

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
h
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6468387

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	

*__inference_conv2d_9_layer_call_fn_6468532

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6466848
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468526:'#
!
_user_specified_name	6468528
Ё
l
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6467759

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
ў
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6468406

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ЮB

R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6466158

inputs/
!reshape_1_readvariableop_resource:@/
!reshape_2_readvariableop_resource:@
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :@d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@Џ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :@h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:@v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:@*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:@T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@i
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@X
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ЮB

R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6466534

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
р
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6468370
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1
р
v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6468506
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1
	

*__inference_conv2d_8_layer_call_fn_6468396

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6466813
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468390:'#
!
_user_specified_name	6468392
р
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6468234
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ q
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1
ЮB

R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6468075

inputs/
!reshape_1_readvariableop_resource: /
!reshape_2_readvariableop_resource: 
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B : d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ Џ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B : h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
: v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
: T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ i
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ {
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Я
V
"__inference__update_step_xla_44471
gradient"
variable:@ *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@ : *
	_noinline(:P L
&
_output_shapes
:@ 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Я
V
"__inference__update_step_xla_44531
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_44406
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	
 
7__inference_group_normalization_4_layer_call_fn_6467884

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6466158
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467878:'#
!
_user_specified_name	6467880
А
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6467846

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44416
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ё
l
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6465998

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_44411
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ё
l
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6465824

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
 
7__inference_group_normalization_8_layer_call_fn_6468415

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6466534
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468409:'#
!
_user_specified_name	6468411
ЮB

R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6468211

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ЮB

R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6467720

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
А
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6467614

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44456
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
з
M
1__inference_up_sampling2d_4_layer_call_fn_6468511

inputs
identityї
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6466555
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Б
ќ
C__inference_conv2d_layer_call_and_return_conditional_losses_6467401

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
	
 
7__inference_group_normalization_7_layer_call_fn_6468279

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6466440
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468273:'#
!
_user_specified_name	6468275
ЬB

P__inference_group_normalization_layer_call_and_return_conditional_losses_6467488

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
	

*__inference_conv2d_2_layer_call_fn_6467623

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6466626
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467617:'#
!
_user_specified_name	6467619
А
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6466615

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44401
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
h
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6466555

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
 
7__inference_group_normalization_3_layer_call_fn_6467768

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6466071
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467762:'#
!
_user_specified_name	6467764
Г
ў
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6467865

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ё
l
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6466085

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6466758

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44526
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
!
Ј	
'__inference_model_layer_call_fn_6467139
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: $

unknown_15: @

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@

unknown_24:

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_6466977
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1:'#
!
_user_specified_name	6467061:'#
!
_user_specified_name	6467063:'#
!
_user_specified_name	6467065:'#
!
_user_specified_name	6467067:'#
!
_user_specified_name	6467069:'#
!
_user_specified_name	6467071:'#
!
_user_specified_name	6467073:'#
!
_user_specified_name	6467075:'	#
!
_user_specified_name	6467077:'
#
!
_user_specified_name	6467079:'#
!
_user_specified_name	6467081:'#
!
_user_specified_name	6467083:'#
!
_user_specified_name	6467085:'#
!
_user_specified_name	6467087:'#
!
_user_specified_name	6467089:'#
!
_user_specified_name	6467091:'#
!
_user_specified_name	6467093:'#
!
_user_specified_name	6467095:'#
!
_user_specified_name	6467097:'#
!
_user_specified_name	6467099:'#
!
_user_specified_name	6467101:'#
!
_user_specified_name	6467103:'#
!
_user_specified_name	6467105:'#
!
_user_specified_name	6467107:'#
!
_user_specified_name	6467109:'#
!
_user_specified_name	6467111:'#
!
_user_specified_name	6467113:'#
!
_user_specified_name	6467115:'#
!
_user_specified_name	6467117:'#
!
_user_specified_name	6467119:'#
!
_user_specified_name	6467121:' #
!
_user_specified_name	6467123:'!#
!
_user_specified_name	6467125:'"#
!
_user_specified_name	6467127:'##
!
_user_specified_name	6467129:'$#
!
_user_specified_name	6467131:'%#
!
_user_specified_name	6467133:'&#
!
_user_specified_name	6467135
Ћ
J
"__inference__update_step_xla_44466
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
E
)__inference_re_lu_8_layer_call_fn_6468488

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6466828z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
ў
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6468134

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ё
l
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6467643

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6468115

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44441
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
E
)__inference_re_lu_1_layer_call_fn_6467609

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6466615z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44481
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Г
ў
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6467749

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Г
ў
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467517

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
А
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6466669

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6465897

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ЁЩ

B__inference_model_layer_call_and_return_conditional_losses_6466977
input_1(
conv2d_6466858:
conv2d_6466860:)
group_normalization_6466864:)
group_normalization_6466866:*
conv2d_1_6466870:
conv2d_1_6466872:+
group_normalization_1_6466876:+
group_normalization_1_6466878:*
conv2d_2_6466882:
conv2d_2_6466884:+
group_normalization_2_6466888:+
group_normalization_2_6466890:*
conv2d_3_6466894: 
conv2d_3_6466896: +
group_normalization_3_6466900: +
group_normalization_3_6466902: *
conv2d_4_6466906: @
conv2d_4_6466908:@+
group_normalization_4_6466912:@+
group_normalization_4_6466914:@*
conv2d_5_6466919:@ 
conv2d_5_6466921: +
group_normalization_5_6466924: +
group_normalization_5_6466926: *
conv2d_6_6466932:@
conv2d_6_6466934:+
group_normalization_6_6466937:+
group_normalization_6_6466939:*
conv2d_7_6466945: 
conv2d_7_6466947:+
group_normalization_7_6466950:+
group_normalization_7_6466952:*
conv2d_8_6466958:
conv2d_8_6466960:+
group_normalization_8_6466963:+
group_normalization_8_6466965:*
conv2d_9_6466971:
conv2d_9_6466973:
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ conv2d_7/StatefulPartitionedCallЂ conv2d_8/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂ+group_normalization/StatefulPartitionedCallЂ-group_normalization_1/StatefulPartitionedCallЂ-group_normalization_2/StatefulPartitionedCallЂ-group_normalization_3/StatefulPartitionedCallЂ-group_normalization_4/StatefulPartitionedCallЂ-group_normalization_5/StatefulPartitionedCallЂ-group_normalization_6/StatefulPartitionedCallЂ-group_normalization_7/StatefulPartitionedCallЂ-group_normalization_8/StatefulPartitionedCallЃ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_6466858conv2d_6466860*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_6466572Ѓ
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6465737њ
+group_normalization/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0group_normalization_6466864group_normalization_6466866*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_group_normalization_layer_call_and_return_conditional_losses_6465810
re_lu/PartitionedCallPartitionedCall4group_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_6466588Т
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_6466870conv2d_1_6466872*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6466599Љ
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6465824
-group_normalization_1/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0group_normalization_1_6466876group_normalization_1_6466878*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6465897
re_lu_1/PartitionedCallPartitionedCall6group_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6466615Ф
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_6466882conv2d_2_6466884*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6466626Љ
#average_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6465911
-group_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_2/PartitionedCall:output:0group_normalization_2_6466888group_normalization_2_6466890*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6465984
re_lu_2/PartitionedCallPartitionedCall6group_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6466642Ф
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_6466894conv2d_3_6466896*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6466653Љ
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6465998
-group_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0group_normalization_3_6466900group_normalization_3_6466902*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6466071
re_lu_3/PartitionedCallPartitionedCall6group_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6466669Ф
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_6466906conv2d_4_6466908*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6466680Љ
#average_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6466085
-group_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_4/PartitionedCall:output:0group_normalization_4_6466912group_normalization_4_6466914*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6466158
re_lu_4/PartitionedCallPartitionedCall6group_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6466696
up_sampling2d/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6466179Ъ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_6466919conv2d_5_6466921*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6466708
-group_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0group_normalization_5_6466924group_normalization_5_6466926*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6466252
re_lu_5/PartitionedCallPartitionedCall6group_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6466723Г
concatenate/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0 re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_6466731
up_sampling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6466273Ь
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_6466932conv2d_6_6466934*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6466743
-group_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0group_normalization_6_6466937group_normalization_6_6466939*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6466346
re_lu_6/PartitionedCallPartitionedCall6group_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6466758З
concatenate_1/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0 re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6466766
up_sampling2d_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6466367Ь
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_6466945conv2d_7_6466947*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6466778
-group_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0group_normalization_7_6466950group_normalization_7_6466952*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6466440
re_lu_7/PartitionedCallPartitionedCall6group_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6466793З
concatenate_2/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0 re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6466801
up_sampling2d_3/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6466461Ь
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_6466958conv2d_8_6466960*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6466813
-group_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0group_normalization_8_6466963group_normalization_8_6466965*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6466534
re_lu_8/PartitionedCallPartitionedCall6group_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6466828Е
concatenate_3/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0 re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6466836
up_sampling2d_4/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6466555Ь
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_9_6466971conv2d_9_6466973*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6466848
IdentityIdentity)conv2d_9/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЌ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall,^group_normalization/StatefulPartitionedCall.^group_normalization_1/StatefulPartitionedCall.^group_normalization_2/StatefulPartitionedCall.^group_normalization_3/StatefulPartitionedCall.^group_normalization_4/StatefulPartitionedCall.^group_normalization_5/StatefulPartitionedCall.^group_normalization_6/StatefulPartitionedCall.^group_normalization_7/StatefulPartitionedCall.^group_normalization_8/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2Z
+group_normalization/StatefulPartitionedCall+group_normalization/StatefulPartitionedCall2^
-group_normalization_1/StatefulPartitionedCall-group_normalization_1/StatefulPartitionedCall2^
-group_normalization_2/StatefulPartitionedCall-group_normalization_2/StatefulPartitionedCall2^
-group_normalization_3/StatefulPartitionedCall-group_normalization_3/StatefulPartitionedCall2^
-group_normalization_4/StatefulPartitionedCall-group_normalization_4/StatefulPartitionedCall2^
-group_normalization_5/StatefulPartitionedCall-group_normalization_5/StatefulPartitionedCall2^
-group_normalization_6/StatefulPartitionedCall-group_normalization_6/StatefulPartitionedCall2^
-group_normalization_7/StatefulPartitionedCall-group_normalization_7/StatefulPartitionedCall2^
-group_normalization_8/StatefulPartitionedCall-group_normalization_8/StatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1:'#
!
_user_specified_name	6466858:'#
!
_user_specified_name	6466860:'#
!
_user_specified_name	6466864:'#
!
_user_specified_name	6466866:'#
!
_user_specified_name	6466870:'#
!
_user_specified_name	6466872:'#
!
_user_specified_name	6466876:'#
!
_user_specified_name	6466878:'	#
!
_user_specified_name	6466882:'
#
!
_user_specified_name	6466884:'#
!
_user_specified_name	6466888:'#
!
_user_specified_name	6466890:'#
!
_user_specified_name	6466894:'#
!
_user_specified_name	6466896:'#
!
_user_specified_name	6466900:'#
!
_user_specified_name	6466902:'#
!
_user_specified_name	6466906:'#
!
_user_specified_name	6466908:'#
!
_user_specified_name	6466912:'#
!
_user_specified_name	6466914:'#
!
_user_specified_name	6466919:'#
!
_user_specified_name	6466921:'#
!
_user_specified_name	6466924:'#
!
_user_specified_name	6466926:'#
!
_user_specified_name	6466932:'#
!
_user_specified_name	6466934:'#
!
_user_specified_name	6466937:'#
!
_user_specified_name	6466939:'#
!
_user_specified_name	6466945:'#
!
_user_specified_name	6466947:'#
!
_user_specified_name	6466950:' #
!
_user_specified_name	6466952:'!#
!
_user_specified_name	6466958:'"#
!
_user_specified_name	6466960:'##
!
_user_specified_name	6466963:'$#
!
_user_specified_name	6466965:'%#
!
_user_specified_name	6466971:'&#
!
_user_specified_name	6466973
Ћ
J
"__inference__update_step_xla_44501
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
А
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6466828

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
t
H__inference_concatenate_layer_call_and_return_conditional_losses_6468098
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@q
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs_1
ќ

(__inference_conv2d_layer_call_fn_6467391

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_6466572
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467385:'#
!
_user_specified_name	6467387
Ђ
E
)__inference_re_lu_6_layer_call_fn_6468216

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6466758z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6466071

inputs/
!reshape_1_readvariableop_resource: /
!reshape_2_readvariableop_resource: 
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B : d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ Џ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B : h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
: v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
: T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ i
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ {
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Г
ў
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6466653

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ
J
"__inference__update_step_xla_44546
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
E
)__inference_re_lu_5_layer_call_fn_6468080

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6466723z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_44551
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ЁЩ

B__inference_model_layer_call_and_return_conditional_losses_6466855
input_1(
conv2d_6466573:
conv2d_6466575:)
group_normalization_6466579:)
group_normalization_6466581:*
conv2d_1_6466600:
conv2d_1_6466602:+
group_normalization_1_6466606:+
group_normalization_1_6466608:*
conv2d_2_6466627:
conv2d_2_6466629:+
group_normalization_2_6466633:+
group_normalization_2_6466635:*
conv2d_3_6466654: 
conv2d_3_6466656: +
group_normalization_3_6466660: +
group_normalization_3_6466662: *
conv2d_4_6466681: @
conv2d_4_6466683:@+
group_normalization_4_6466687:@+
group_normalization_4_6466689:@*
conv2d_5_6466709:@ 
conv2d_5_6466711: +
group_normalization_5_6466714: +
group_normalization_5_6466716: *
conv2d_6_6466744:@
conv2d_6_6466746:+
group_normalization_6_6466749:+
group_normalization_6_6466751:*
conv2d_7_6466779: 
conv2d_7_6466781:+
group_normalization_7_6466784:+
group_normalization_7_6466786:*
conv2d_8_6466814:
conv2d_8_6466816:+
group_normalization_8_6466819:+
group_normalization_8_6466821:*
conv2d_9_6466849:
conv2d_9_6466851:
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ conv2d_7/StatefulPartitionedCallЂ conv2d_8/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂ+group_normalization/StatefulPartitionedCallЂ-group_normalization_1/StatefulPartitionedCallЂ-group_normalization_2/StatefulPartitionedCallЂ-group_normalization_3/StatefulPartitionedCallЂ-group_normalization_4/StatefulPartitionedCallЂ-group_normalization_5/StatefulPartitionedCallЂ-group_normalization_6/StatefulPartitionedCallЂ-group_normalization_7/StatefulPartitionedCallЂ-group_normalization_8/StatefulPartitionedCallЃ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_6466573conv2d_6466575*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_6466572Ѓ
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6465737њ
+group_normalization/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0group_normalization_6466579group_normalization_6466581*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_group_normalization_layer_call_and_return_conditional_losses_6465810
re_lu/PartitionedCallPartitionedCall4group_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_6466588Т
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_6466600conv2d_1_6466602*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6466599Љ
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6465824
-group_normalization_1/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0group_normalization_1_6466606group_normalization_1_6466608*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6465897
re_lu_1/PartitionedCallPartitionedCall6group_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6466615Ф
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_6466627conv2d_2_6466629*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6466626Љ
#average_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6465911
-group_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_2/PartitionedCall:output:0group_normalization_2_6466633group_normalization_2_6466635*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6465984
re_lu_2/PartitionedCallPartitionedCall6group_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6466642Ф
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_6466654conv2d_3_6466656*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6466653Љ
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6465998
-group_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0group_normalization_3_6466660group_normalization_3_6466662*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6466071
re_lu_3/PartitionedCallPartitionedCall6group_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6466669Ф
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_6466681conv2d_4_6466683*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6466680Љ
#average_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6466085
-group_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_4/PartitionedCall:output:0group_normalization_4_6466687group_normalization_4_6466689*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6466158
re_lu_4/PartitionedCallPartitionedCall6group_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6466696
up_sampling2d/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6466179Ъ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_6466709conv2d_5_6466711*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6466708
-group_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0group_normalization_5_6466714group_normalization_5_6466716*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6466252
re_lu_5/PartitionedCallPartitionedCall6group_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6466723Г
concatenate/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0 re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_6466731
up_sampling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6466273Ь
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_6466744conv2d_6_6466746*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6466743
-group_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0group_normalization_6_6466749group_normalization_6_6466751*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6466346
re_lu_6/PartitionedCallPartitionedCall6group_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6466758З
concatenate_1/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0 re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6466766
up_sampling2d_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6466367Ь
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_6466779conv2d_7_6466781*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6466778
-group_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0group_normalization_7_6466784group_normalization_7_6466786*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6466440
re_lu_7/PartitionedCallPartitionedCall6group_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6466793З
concatenate_2/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0 re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6466801
up_sampling2d_3/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6466461Ь
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_6466814conv2d_8_6466816*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6466813
-group_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0group_normalization_8_6466819group_normalization_8_6466821*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6466534
re_lu_8/PartitionedCallPartitionedCall6group_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6466828Е
concatenate_3/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0 re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6466836
up_sampling2d_4/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6466555Ь
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_9_6466849conv2d_9_6466851*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6466848
IdentityIdentity)conv2d_9/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЌ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall,^group_normalization/StatefulPartitionedCall.^group_normalization_1/StatefulPartitionedCall.^group_normalization_2/StatefulPartitionedCall.^group_normalization_3/StatefulPartitionedCall.^group_normalization_4/StatefulPartitionedCall.^group_normalization_5/StatefulPartitionedCall.^group_normalization_6/StatefulPartitionedCall.^group_normalization_7/StatefulPartitionedCall.^group_normalization_8/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2Z
+group_normalization/StatefulPartitionedCall+group_normalization/StatefulPartitionedCall2^
-group_normalization_1/StatefulPartitionedCall-group_normalization_1/StatefulPartitionedCall2^
-group_normalization_2/StatefulPartitionedCall-group_normalization_2/StatefulPartitionedCall2^
-group_normalization_3/StatefulPartitionedCall-group_normalization_3/StatefulPartitionedCall2^
-group_normalization_4/StatefulPartitionedCall-group_normalization_4/StatefulPartitionedCall2^
-group_normalization_5/StatefulPartitionedCall-group_normalization_5/StatefulPartitionedCall2^
-group_normalization_6/StatefulPartitionedCall-group_normalization_6/StatefulPartitionedCall2^
-group_normalization_7/StatefulPartitionedCall-group_normalization_7/StatefulPartitionedCall2^
-group_normalization_8/StatefulPartitionedCall-group_normalization_8/StatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1:'#
!
_user_specified_name	6466573:'#
!
_user_specified_name	6466575:'#
!
_user_specified_name	6466579:'#
!
_user_specified_name	6466581:'#
!
_user_specified_name	6466600:'#
!
_user_specified_name	6466602:'#
!
_user_specified_name	6466606:'#
!
_user_specified_name	6466608:'	#
!
_user_specified_name	6466627:'
#
!
_user_specified_name	6466629:'#
!
_user_specified_name	6466633:'#
!
_user_specified_name	6466635:'#
!
_user_specified_name	6466654:'#
!
_user_specified_name	6466656:'#
!
_user_specified_name	6466660:'#
!
_user_specified_name	6466662:'#
!
_user_specified_name	6466681:'#
!
_user_specified_name	6466683:'#
!
_user_specified_name	6466687:'#
!
_user_specified_name	6466689:'#
!
_user_specified_name	6466709:'#
!
_user_specified_name	6466711:'#
!
_user_specified_name	6466714:'#
!
_user_specified_name	6466716:'#
!
_user_specified_name	6466744:'#
!
_user_specified_name	6466746:'#
!
_user_specified_name	6466749:'#
!
_user_specified_name	6466751:'#
!
_user_specified_name	6466779:'#
!
_user_specified_name	6466781:'#
!
_user_specified_name	6466784:' #
!
_user_specified_name	6466786:'!#
!
_user_specified_name	6466814:'"#
!
_user_specified_name	6466816:'##
!
_user_specified_name	6466819:'$#
!
_user_specified_name	6466821:'%#
!
_user_specified_name	6466849:'&#
!
_user_specified_name	6466851
у
Y
-__inference_concatenate_layer_call_fn_6468091
inputs_0
inputs_1
identityї
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_6466731z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs_1
ч
[
/__inference_concatenate_1_layer_call_fn_6468227
inputs_0
inputs_1
identityљ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6466766z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1
п
Q
5__inference_average_pooling2d_2_layer_call_fn_6467638

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6465911
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6465984

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Я
V
"__inference__update_step_xla_44371
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
з
M
1__inference_up_sampling2d_2_layer_call_fn_6468239

inputs
identityї
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6466367
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6467952

inputs/
!reshape_1_readvariableop_resource:@/
!reshape_2_readvariableop_resource:@
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :@d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@Џ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :@h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:@v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:@*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:@T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@i
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@X
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
 
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6466179

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44381
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
и
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6466766

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ q
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
`
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6468085

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
!
Ј	
'__inference_model_layer_call_fn_6467058
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: $

unknown_15: @

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@

unknown_24:

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_6466855
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1:'#
!
_user_specified_name	6466980:'#
!
_user_specified_name	6466982:'#
!
_user_specified_name	6466984:'#
!
_user_specified_name	6466986:'#
!
_user_specified_name	6466988:'#
!
_user_specified_name	6466990:'#
!
_user_specified_name	6466992:'#
!
_user_specified_name	6466994:'	#
!
_user_specified_name	6466996:'
#
!
_user_specified_name	6466998:'#
!
_user_specified_name	6467000:'#
!
_user_specified_name	6467002:'#
!
_user_specified_name	6467004:'#
!
_user_specified_name	6467006:'#
!
_user_specified_name	6467008:'#
!
_user_specified_name	6467010:'#
!
_user_specified_name	6467012:'#
!
_user_specified_name	6467014:'#
!
_user_specified_name	6467016:'#
!
_user_specified_name	6467018:'#
!
_user_specified_name	6467020:'#
!
_user_specified_name	6467022:'#
!
_user_specified_name	6467024:'#
!
_user_specified_name	6467026:'#
!
_user_specified_name	6467028:'#
!
_user_specified_name	6467030:'#
!
_user_specified_name	6467032:'#
!
_user_specified_name	6467034:'#
!
_user_specified_name	6467036:'#
!
_user_specified_name	6467038:'#
!
_user_specified_name	6467040:' #
!
_user_specified_name	6467042:'!#
!
_user_specified_name	6467044:'"#
!
_user_specified_name	6467046:'##
!
_user_specified_name	6467048:'$#
!
_user_specified_name	6467050:'%#
!
_user_specified_name	6467052:'&#
!
_user_specified_name	6467054
	
 
7__inference_group_normalization_5_layer_call_fn_6468007

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6466252
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468001:'#
!
_user_specified_name	6468003
	

*__inference_conv2d_5_layer_call_fn_6467988

inputs!
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6466708
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467982:'#
!
_user_specified_name	6467984
ч
[
/__inference_concatenate_3_layer_call_fn_6468499
inputs_0
inputs_1
identityљ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6466836z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1
А
`
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6466723

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ч
[
/__inference_concatenate_2_layer_call_fn_6468363
inputs_0
inputs_1
identityљ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6466801z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1
Ђ
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6466367

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
`
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6467962

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Г
ў
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6467633

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
и
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6466801

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6468221

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6468483

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ЮB

R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6467836

inputs/
!reshape_1_readvariableop_resource: /
!reshape_2_readvariableop_resource: 
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B : d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ Џ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B : h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
: v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
: T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ i
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ {
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
h
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6468523

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6465737

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44461
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
А
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6466642

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44476
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ЮB

R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6466252

inputs/
!reshape_1_readvariableop_resource: /
!reshape_2_readvariableop_resource: 
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B : d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ Џ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B : h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
: v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
: *
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
: T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ i
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ {
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ {
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
л
O
3__inference_average_pooling2d_layer_call_fn_6467406

inputs
identityљ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6465737
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Q
5__inference_average_pooling2d_1_layer_call_fn_6467522

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6465824
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
E
)__inference_re_lu_2_layer_call_fn_6467725

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6466642z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
кЇ
цN
#__inference__traced_restore_6469635
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias::
,assignvariableop_2_group_normalization_gamma:9
+assignvariableop_3_group_normalization_beta:<
"assignvariableop_4_conv2d_1_kernel:.
 assignvariableop_5_conv2d_1_bias:<
.assignvariableop_6_group_normalization_1_gamma:;
-assignvariableop_7_group_normalization_1_beta:<
"assignvariableop_8_conv2d_2_kernel:.
 assignvariableop_9_conv2d_2_bias:=
/assignvariableop_10_group_normalization_2_gamma:<
.assignvariableop_11_group_normalization_2_beta:=
#assignvariableop_12_conv2d_3_kernel: /
!assignvariableop_13_conv2d_3_bias: =
/assignvariableop_14_group_normalization_3_gamma: <
.assignvariableop_15_group_normalization_3_beta: =
#assignvariableop_16_conv2d_4_kernel: @/
!assignvariableop_17_conv2d_4_bias:@=
/assignvariableop_18_group_normalization_4_gamma:@<
.assignvariableop_19_group_normalization_4_beta:@=
#assignvariableop_20_conv2d_5_kernel:@ /
!assignvariableop_21_conv2d_5_bias: =
/assignvariableop_22_group_normalization_5_gamma: <
.assignvariableop_23_group_normalization_5_beta: =
#assignvariableop_24_conv2d_6_kernel:@/
!assignvariableop_25_conv2d_6_bias:=
/assignvariableop_26_group_normalization_6_gamma:<
.assignvariableop_27_group_normalization_6_beta:=
#assignvariableop_28_conv2d_7_kernel: /
!assignvariableop_29_conv2d_7_bias:=
/assignvariableop_30_group_normalization_7_gamma:<
.assignvariableop_31_group_normalization_7_beta:=
#assignvariableop_32_conv2d_8_kernel:/
!assignvariableop_33_conv2d_8_bias:=
/assignvariableop_34_group_normalization_8_gamma:<
.assignvariableop_35_group_normalization_8_beta:=
#assignvariableop_36_conv2d_9_kernel:/
!assignvariableop_37_conv2d_9_bias:'
assignvariableop_38_iteration:	 +
!assignvariableop_39_learning_rate: B
(assignvariableop_40_adam_m_conv2d_kernel:B
(assignvariableop_41_adam_v_conv2d_kernel:4
&assignvariableop_42_adam_m_conv2d_bias:4
&assignvariableop_43_adam_v_conv2d_bias:B
4assignvariableop_44_adam_m_group_normalization_gamma:B
4assignvariableop_45_adam_v_group_normalization_gamma:A
3assignvariableop_46_adam_m_group_normalization_beta:A
3assignvariableop_47_adam_v_group_normalization_beta:D
*assignvariableop_48_adam_m_conv2d_1_kernel:D
*assignvariableop_49_adam_v_conv2d_1_kernel:6
(assignvariableop_50_adam_m_conv2d_1_bias:6
(assignvariableop_51_adam_v_conv2d_1_bias:D
6assignvariableop_52_adam_m_group_normalization_1_gamma:D
6assignvariableop_53_adam_v_group_normalization_1_gamma:C
5assignvariableop_54_adam_m_group_normalization_1_beta:C
5assignvariableop_55_adam_v_group_normalization_1_beta:D
*assignvariableop_56_adam_m_conv2d_2_kernel:D
*assignvariableop_57_adam_v_conv2d_2_kernel:6
(assignvariableop_58_adam_m_conv2d_2_bias:6
(assignvariableop_59_adam_v_conv2d_2_bias:D
6assignvariableop_60_adam_m_group_normalization_2_gamma:D
6assignvariableop_61_adam_v_group_normalization_2_gamma:C
5assignvariableop_62_adam_m_group_normalization_2_beta:C
5assignvariableop_63_adam_v_group_normalization_2_beta:D
*assignvariableop_64_adam_m_conv2d_3_kernel: D
*assignvariableop_65_adam_v_conv2d_3_kernel: 6
(assignvariableop_66_adam_m_conv2d_3_bias: 6
(assignvariableop_67_adam_v_conv2d_3_bias: D
6assignvariableop_68_adam_m_group_normalization_3_gamma: D
6assignvariableop_69_adam_v_group_normalization_3_gamma: C
5assignvariableop_70_adam_m_group_normalization_3_beta: C
5assignvariableop_71_adam_v_group_normalization_3_beta: D
*assignvariableop_72_adam_m_conv2d_4_kernel: @D
*assignvariableop_73_adam_v_conv2d_4_kernel: @6
(assignvariableop_74_adam_m_conv2d_4_bias:@6
(assignvariableop_75_adam_v_conv2d_4_bias:@D
6assignvariableop_76_adam_m_group_normalization_4_gamma:@D
6assignvariableop_77_adam_v_group_normalization_4_gamma:@C
5assignvariableop_78_adam_m_group_normalization_4_beta:@C
5assignvariableop_79_adam_v_group_normalization_4_beta:@D
*assignvariableop_80_adam_m_conv2d_5_kernel:@ D
*assignvariableop_81_adam_v_conv2d_5_kernel:@ 6
(assignvariableop_82_adam_m_conv2d_5_bias: 6
(assignvariableop_83_adam_v_conv2d_5_bias: D
6assignvariableop_84_adam_m_group_normalization_5_gamma: D
6assignvariableop_85_adam_v_group_normalization_5_gamma: C
5assignvariableop_86_adam_m_group_normalization_5_beta: C
5assignvariableop_87_adam_v_group_normalization_5_beta: D
*assignvariableop_88_adam_m_conv2d_6_kernel:@D
*assignvariableop_89_adam_v_conv2d_6_kernel:@6
(assignvariableop_90_adam_m_conv2d_6_bias:6
(assignvariableop_91_adam_v_conv2d_6_bias:D
6assignvariableop_92_adam_m_group_normalization_6_gamma:D
6assignvariableop_93_adam_v_group_normalization_6_gamma:C
5assignvariableop_94_adam_m_group_normalization_6_beta:C
5assignvariableop_95_adam_v_group_normalization_6_beta:D
*assignvariableop_96_adam_m_conv2d_7_kernel: D
*assignvariableop_97_adam_v_conv2d_7_kernel: 6
(assignvariableop_98_adam_m_conv2d_7_bias:6
(assignvariableop_99_adam_v_conv2d_7_bias:E
7assignvariableop_100_adam_m_group_normalization_7_gamma:E
7assignvariableop_101_adam_v_group_normalization_7_gamma:D
6assignvariableop_102_adam_m_group_normalization_7_beta:D
6assignvariableop_103_adam_v_group_normalization_7_beta:E
+assignvariableop_104_adam_m_conv2d_8_kernel:E
+assignvariableop_105_adam_v_conv2d_8_kernel:7
)assignvariableop_106_adam_m_conv2d_8_bias:7
)assignvariableop_107_adam_v_conv2d_8_bias:E
7assignvariableop_108_adam_m_group_normalization_8_gamma:E
7assignvariableop_109_adam_v_group_normalization_8_gamma:D
6assignvariableop_110_adam_m_group_normalization_8_beta:D
6assignvariableop_111_adam_v_group_normalization_8_beta:E
+assignvariableop_112_adam_m_conv2d_9_kernel:E
+assignvariableop_113_adam_v_conv2d_9_kernel:7
)assignvariableop_114_adam_m_conv2d_9_bias:7
)assignvariableop_115_adam_v_conv2d_9_bias:$
assignvariableop_116_total: $
assignvariableop_117_count: 
identity_119ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_101ЂAssignVariableOp_102ЂAssignVariableOp_103ЂAssignVariableOp_104ЂAssignVariableOp_105ЂAssignVariableOp_106ЂAssignVariableOp_107ЂAssignVariableOp_108ЂAssignVariableOp_109ЂAssignVariableOp_11ЂAssignVariableOp_110ЂAssignVariableOp_111ЂAssignVariableOp_112ЂAssignVariableOp_113ЂAssignVariableOp_114ЂAssignVariableOp_115ЂAssignVariableOp_116ЂAssignVariableOp_117ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_99њ1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:w*
dtype0* 1
value1B1wB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHс
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*
valueљBіwB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ѕ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes{
y2w	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp,assignvariableop_2_group_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_3AssignVariableOp+assignvariableop_3_group_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_6AssignVariableOp.assignvariableop_6_group_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_7AssignVariableOp-assignvariableop_7_group_normalization_1_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_10AssignVariableOp/assignvariableop_10_group_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_11AssignVariableOp.assignvariableop_11_group_normalization_2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_14AssignVariableOp/assignvariableop_14_group_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_15AssignVariableOp.assignvariableop_15_group_normalization_3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_4_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_4_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_18AssignVariableOp/assignvariableop_18_group_normalization_4_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_19AssignVariableOp.assignvariableop_19_group_normalization_4_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_5_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_5_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_22AssignVariableOp/assignvariableop_22_group_normalization_5_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_23AssignVariableOp.assignvariableop_23_group_normalization_5_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_6_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_6_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_26AssignVariableOp/assignvariableop_26_group_normalization_6_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_27AssignVariableOp.assignvariableop_27_group_normalization_6_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_7_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_7_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_30AssignVariableOp/assignvariableop_30_group_normalization_7_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_31AssignVariableOp.assignvariableop_31_group_normalization_7_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv2d_8_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_33AssignVariableOp!assignvariableop_33_conv2d_8_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_34AssignVariableOp/assignvariableop_34_group_normalization_8_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_35AssignVariableOp.assignvariableop_35_group_normalization_8_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_9_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_9_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_38AssignVariableOpassignvariableop_38_iterationIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_39AssignVariableOp!assignvariableop_39_learning_rateIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_conv2d_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_conv2d_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_m_conv2d_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_43AssignVariableOp&assignvariableop_43_adam_v_conv2d_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_m_group_normalization_gammaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_v_group_normalization_gammaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_m_group_normalization_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_47AssignVariableOp3assignvariableop_47_adam_v_group_normalization_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_conv2d_1_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_conv2d_1_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_conv2d_1_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_conv2d_1_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_m_group_normalization_1_gammaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_v_group_normalization_1_gammaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_m_group_normalization_1_betaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_v_group_normalization_1_betaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_m_conv2d_2_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_v_conv2d_2_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_m_conv2d_2_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_v_conv2d_2_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_m_group_normalization_2_gammaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_v_group_normalization_2_gammaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_m_group_normalization_2_betaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_63AssignVariableOp5assignvariableop_63_adam_v_group_normalization_2_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_m_conv2d_3_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_v_conv2d_3_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_m_conv2d_3_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_v_conv2d_3_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_m_group_normalization_3_gammaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_v_group_normalization_3_gammaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_m_group_normalization_3_betaIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_71AssignVariableOp5assignvariableop_71_adam_v_group_normalization_3_betaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_m_conv2d_4_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_v_conv2d_4_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_m_conv2d_4_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_v_conv2d_4_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_m_group_normalization_4_gammaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_v_group_normalization_4_gammaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_78AssignVariableOp5assignvariableop_78_adam_m_group_normalization_4_betaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_79AssignVariableOp5assignvariableop_79_adam_v_group_normalization_4_betaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_m_conv2d_5_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_v_conv2d_5_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_m_conv2d_5_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_v_conv2d_5_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_m_group_normalization_5_gammaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_v_group_normalization_5_gammaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_m_group_normalization_5_betaIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_87AssignVariableOp5assignvariableop_87_adam_v_group_normalization_5_betaIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_88AssignVariableOp*assignvariableop_88_adam_m_conv2d_6_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_v_conv2d_6_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_m_conv2d_6_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_v_conv2d_6_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_m_group_normalization_6_gammaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_v_group_normalization_6_gammaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_m_group_normalization_6_betaIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_95AssignVariableOp5assignvariableop_95_adam_v_group_normalization_6_betaIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_m_conv2d_7_kernelIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_v_conv2d_7_kernelIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_m_conv2d_7_biasIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_v_conv2d_7_biasIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_m_group_normalization_7_gammaIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_v_group_normalization_7_gammaIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_m_group_normalization_7_betaIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_103AssignVariableOp6assignvariableop_103_adam_v_group_normalization_7_betaIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_104AssignVariableOp+assignvariableop_104_adam_m_conv2d_8_kernelIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_v_conv2d_8_kernelIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_m_conv2d_8_biasIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_107AssignVariableOp)assignvariableop_107_adam_v_conv2d_8_biasIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_m_group_normalization_8_gammaIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_v_group_normalization_8_gammaIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_110AssignVariableOp6assignvariableop_110_adam_m_group_normalization_8_betaIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_111AssignVariableOp6assignvariableop_111_adam_v_group_normalization_8_betaIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_112AssignVariableOp+assignvariableop_112_adam_m_conv2d_9_kernelIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_113AssignVariableOp+assignvariableop_113_adam_v_conv2d_9_kernelIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_114AssignVariableOp)assignvariableop_114_adam_m_conv2d_9_biasIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_115AssignVariableOp)assignvariableop_115_adam_v_conv2d_9_biasIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_116AssignVariableOpassignvariableop_116_totalIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_117AssignVariableOpassignvariableop_117_countIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_118Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_119IdentityIdentity_118:output:0^NoOp_1*
T0*
_output_shapes
: Ю
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_119Identity_119:output:0*(
_construction_contextkEagerRuntime*
_input_shapesё
ю: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv2d/kernel:+'
%
_user_specified_nameconv2d/bias:95
3
_user_specified_namegroup_normalization/gamma:84
2
_user_specified_namegroup_normalization/beta:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv2d_1/bias:;7
5
_user_specified_namegroup_normalization_1/gamma::6
4
_user_specified_namegroup_normalization_1/beta:/	+
)
_user_specified_nameconv2d_2/kernel:-
)
'
_user_specified_nameconv2d_2/bias:;7
5
_user_specified_namegroup_normalization_2/gamma::6
4
_user_specified_namegroup_normalization_2/beta:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_3/bias:;7
5
_user_specified_namegroup_normalization_3/gamma::6
4
_user_specified_namegroup_normalization_3/beta:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:;7
5
_user_specified_namegroup_normalization_4/gamma::6
4
_user_specified_namegroup_normalization_4/beta:/+
)
_user_specified_nameconv2d_5/kernel:-)
'
_user_specified_nameconv2d_5/bias:;7
5
_user_specified_namegroup_normalization_5/gamma::6
4
_user_specified_namegroup_normalization_5/beta:/+
)
_user_specified_nameconv2d_6/kernel:-)
'
_user_specified_nameconv2d_6/bias:;7
5
_user_specified_namegroup_normalization_6/gamma::6
4
_user_specified_namegroup_normalization_6/beta:/+
)
_user_specified_nameconv2d_7/kernel:-)
'
_user_specified_nameconv2d_7/bias:;7
5
_user_specified_namegroup_normalization_7/gamma:: 6
4
_user_specified_namegroup_normalization_7/beta:/!+
)
_user_specified_nameconv2d_8/kernel:-")
'
_user_specified_nameconv2d_8/bias:;#7
5
_user_specified_namegroup_normalization_8/gamma::$6
4
_user_specified_namegroup_normalization_8/beta:/%+
)
_user_specified_nameconv2d_9/kernel:-&)
'
_user_specified_nameconv2d_9/bias:)'%
#
_user_specified_name	iteration:-()
'
_user_specified_namelearning_rate:4)0
.
_user_specified_nameAdam/m/conv2d/kernel:4*0
.
_user_specified_nameAdam/v/conv2d/kernel:2+.
,
_user_specified_nameAdam/m/conv2d/bias:2,.
,
_user_specified_nameAdam/v/conv2d/bias:@-<
:
_user_specified_name" Adam/m/group_normalization/gamma:@.<
:
_user_specified_name" Adam/v/group_normalization/gamma:?/;
9
_user_specified_name!Adam/m/group_normalization/beta:?0;
9
_user_specified_name!Adam/v/group_normalization/beta:612
0
_user_specified_nameAdam/m/conv2d_1/kernel:622
0
_user_specified_nameAdam/v/conv2d_1/kernel:430
.
_user_specified_nameAdam/m/conv2d_1/bias:440
.
_user_specified_nameAdam/v/conv2d_1/bias:B5>
<
_user_specified_name$"Adam/m/group_normalization_1/gamma:B6>
<
_user_specified_name$"Adam/v/group_normalization_1/gamma:A7=
;
_user_specified_name#!Adam/m/group_normalization_1/beta:A8=
;
_user_specified_name#!Adam/v/group_normalization_1/beta:692
0
_user_specified_nameAdam/m/conv2d_2/kernel:6:2
0
_user_specified_nameAdam/v/conv2d_2/kernel:4;0
.
_user_specified_nameAdam/m/conv2d_2/bias:4<0
.
_user_specified_nameAdam/v/conv2d_2/bias:B=>
<
_user_specified_name$"Adam/m/group_normalization_2/gamma:B>>
<
_user_specified_name$"Adam/v/group_normalization_2/gamma:A?=
;
_user_specified_name#!Adam/m/group_normalization_2/beta:A@=
;
_user_specified_name#!Adam/v/group_normalization_2/beta:6A2
0
_user_specified_nameAdam/m/conv2d_3/kernel:6B2
0
_user_specified_nameAdam/v/conv2d_3/kernel:4C0
.
_user_specified_nameAdam/m/conv2d_3/bias:4D0
.
_user_specified_nameAdam/v/conv2d_3/bias:BE>
<
_user_specified_name$"Adam/m/group_normalization_3/gamma:BF>
<
_user_specified_name$"Adam/v/group_normalization_3/gamma:AG=
;
_user_specified_name#!Adam/m/group_normalization_3/beta:AH=
;
_user_specified_name#!Adam/v/group_normalization_3/beta:6I2
0
_user_specified_nameAdam/m/conv2d_4/kernel:6J2
0
_user_specified_nameAdam/v/conv2d_4/kernel:4K0
.
_user_specified_nameAdam/m/conv2d_4/bias:4L0
.
_user_specified_nameAdam/v/conv2d_4/bias:BM>
<
_user_specified_name$"Adam/m/group_normalization_4/gamma:BN>
<
_user_specified_name$"Adam/v/group_normalization_4/gamma:AO=
;
_user_specified_name#!Adam/m/group_normalization_4/beta:AP=
;
_user_specified_name#!Adam/v/group_normalization_4/beta:6Q2
0
_user_specified_nameAdam/m/conv2d_5/kernel:6R2
0
_user_specified_nameAdam/v/conv2d_5/kernel:4S0
.
_user_specified_nameAdam/m/conv2d_5/bias:4T0
.
_user_specified_nameAdam/v/conv2d_5/bias:BU>
<
_user_specified_name$"Adam/m/group_normalization_5/gamma:BV>
<
_user_specified_name$"Adam/v/group_normalization_5/gamma:AW=
;
_user_specified_name#!Adam/m/group_normalization_5/beta:AX=
;
_user_specified_name#!Adam/v/group_normalization_5/beta:6Y2
0
_user_specified_nameAdam/m/conv2d_6/kernel:6Z2
0
_user_specified_nameAdam/v/conv2d_6/kernel:4[0
.
_user_specified_nameAdam/m/conv2d_6/bias:4\0
.
_user_specified_nameAdam/v/conv2d_6/bias:B]>
<
_user_specified_name$"Adam/m/group_normalization_6/gamma:B^>
<
_user_specified_name$"Adam/v/group_normalization_6/gamma:A_=
;
_user_specified_name#!Adam/m/group_normalization_6/beta:A`=
;
_user_specified_name#!Adam/v/group_normalization_6/beta:6a2
0
_user_specified_nameAdam/m/conv2d_7/kernel:6b2
0
_user_specified_nameAdam/v/conv2d_7/kernel:4c0
.
_user_specified_nameAdam/m/conv2d_7/bias:4d0
.
_user_specified_nameAdam/v/conv2d_7/bias:Be>
<
_user_specified_name$"Adam/m/group_normalization_7/gamma:Bf>
<
_user_specified_name$"Adam/v/group_normalization_7/gamma:Ag=
;
_user_specified_name#!Adam/m/group_normalization_7/beta:Ah=
;
_user_specified_name#!Adam/v/group_normalization_7/beta:6i2
0
_user_specified_nameAdam/m/conv2d_8/kernel:6j2
0
_user_specified_nameAdam/v/conv2d_8/kernel:4k0
.
_user_specified_nameAdam/m/conv2d_8/bias:4l0
.
_user_specified_nameAdam/v/conv2d_8/bias:Bm>
<
_user_specified_name$"Adam/m/group_normalization_8/gamma:Bn>
<
_user_specified_name$"Adam/v/group_normalization_8/gamma:Ao=
;
_user_specified_name#!Adam/m/group_normalization_8/beta:Ap=
;
_user_specified_name#!Adam/v/group_normalization_8/beta:6q2
0
_user_specified_nameAdam/m/conv2d_9/kernel:6r2
0
_user_specified_nameAdam/v/conv2d_9/kernel:4s0
.
_user_specified_nameAdam/m/conv2d_9/bias:4t0
.
_user_specified_nameAdam/v/conv2d_9/bias:%u!

_user_specified_nametotal:%v!

_user_specified_namecount
Г
ў
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6468270

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
є
ж#
"__inference__wrapped_model_6465732
input_1E
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:I
;model_group_normalization_reshape_1_readvariableop_resource:I
;model_group_normalization_reshape_2_readvariableop_resource:G
-model_conv2d_1_conv2d_readvariableop_resource:<
.model_conv2d_1_biasadd_readvariableop_resource:K
=model_group_normalization_1_reshape_1_readvariableop_resource:K
=model_group_normalization_1_reshape_2_readvariableop_resource:G
-model_conv2d_2_conv2d_readvariableop_resource:<
.model_conv2d_2_biasadd_readvariableop_resource:K
=model_group_normalization_2_reshape_1_readvariableop_resource:K
=model_group_normalization_2_reshape_2_readvariableop_resource:G
-model_conv2d_3_conv2d_readvariableop_resource: <
.model_conv2d_3_biasadd_readvariableop_resource: K
=model_group_normalization_3_reshape_1_readvariableop_resource: K
=model_group_normalization_3_reshape_2_readvariableop_resource: G
-model_conv2d_4_conv2d_readvariableop_resource: @<
.model_conv2d_4_biasadd_readvariableop_resource:@K
=model_group_normalization_4_reshape_1_readvariableop_resource:@K
=model_group_normalization_4_reshape_2_readvariableop_resource:@G
-model_conv2d_5_conv2d_readvariableop_resource:@ <
.model_conv2d_5_biasadd_readvariableop_resource: K
=model_group_normalization_5_reshape_1_readvariableop_resource: K
=model_group_normalization_5_reshape_2_readvariableop_resource: G
-model_conv2d_6_conv2d_readvariableop_resource:@<
.model_conv2d_6_biasadd_readvariableop_resource:K
=model_group_normalization_6_reshape_1_readvariableop_resource:K
=model_group_normalization_6_reshape_2_readvariableop_resource:G
-model_conv2d_7_conv2d_readvariableop_resource: <
.model_conv2d_7_biasadd_readvariableop_resource:K
=model_group_normalization_7_reshape_1_readvariableop_resource:K
=model_group_normalization_7_reshape_2_readvariableop_resource:G
-model_conv2d_8_conv2d_readvariableop_resource:<
.model_conv2d_8_biasadd_readvariableop_resource:K
=model_group_normalization_8_reshape_1_readvariableop_resource:K
=model_group_normalization_8_reshape_2_readvariableop_resource:G
-model_conv2d_9_conv2d_readvariableop_resource:<
.model_conv2d_9_biasadd_readvariableop_resource:
identityЂ#model/conv2d/BiasAdd/ReadVariableOpЂ"model/conv2d/Conv2D/ReadVariableOpЂ%model/conv2d_1/BiasAdd/ReadVariableOpЂ$model/conv2d_1/Conv2D/ReadVariableOpЂ%model/conv2d_2/BiasAdd/ReadVariableOpЂ$model/conv2d_2/Conv2D/ReadVariableOpЂ%model/conv2d_3/BiasAdd/ReadVariableOpЂ$model/conv2d_3/Conv2D/ReadVariableOpЂ%model/conv2d_4/BiasAdd/ReadVariableOpЂ$model/conv2d_4/Conv2D/ReadVariableOpЂ%model/conv2d_5/BiasAdd/ReadVariableOpЂ$model/conv2d_5/Conv2D/ReadVariableOpЂ%model/conv2d_6/BiasAdd/ReadVariableOpЂ$model/conv2d_6/Conv2D/ReadVariableOpЂ%model/conv2d_7/BiasAdd/ReadVariableOpЂ$model/conv2d_7/Conv2D/ReadVariableOpЂ%model/conv2d_8/BiasAdd/ReadVariableOpЂ$model/conv2d_8/Conv2D/ReadVariableOpЂ%model/conv2d_9/BiasAdd/ReadVariableOpЂ$model/conv2d_9/Conv2D/ReadVariableOpЂ2model/group_normalization/Reshape_1/ReadVariableOpЂ2model/group_normalization/Reshape_2/ReadVariableOpЂ4model/group_normalization_1/Reshape_1/ReadVariableOpЂ4model/group_normalization_1/Reshape_2/ReadVariableOpЂ4model/group_normalization_2/Reshape_1/ReadVariableOpЂ4model/group_normalization_2/Reshape_2/ReadVariableOpЂ4model/group_normalization_3/Reshape_1/ReadVariableOpЂ4model/group_normalization_3/Reshape_2/ReadVariableOpЂ4model/group_normalization_4/Reshape_1/ReadVariableOpЂ4model/group_normalization_4/Reshape_2/ReadVariableOpЂ4model/group_normalization_5/Reshape_1/ReadVariableOpЂ4model/group_normalization_5/Reshape_2/ReadVariableOpЂ4model/group_normalization_6/Reshape_1/ReadVariableOpЂ4model/group_normalization_6/Reshape_2/ReadVariableOpЂ4model/group_normalization_7/Reshape_1/ReadVariableOpЂ4model/group_normalization_7/Reshape_2/ReadVariableOpЂ4model/group_normalization_8/Reshape_1/ReadVariableOpЂ4model/group_normalization_8/Reshape_2/ReadVariableOp
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ц
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџб
model/average_pooling2d/AvgPoolAvgPoolmodel/conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides

model/group_normalization/ShapeShape(model/average_pooling2d/AvgPool:output:0*
T0*
_output_shapes
::эЯ
!model/group_normalization/Shape_1Shape(model/average_pooling2d/AvgPool:output:0*
T0*
_output_shapes
::эЯw
-model/group_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/group_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/group_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
'model/group_normalization/strided_sliceStridedSlice*model/group_normalization/Shape_1:output:06model/group_normalization/strided_slice/stack:output:08model/group_normalization/strided_slice/stack_1:output:08model/group_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
/model/group_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/group_normalization/strided_slice_1StridedSlice*model/group_normalization/Shape_1:output:08model/group_normalization/strided_slice_1/stack:output:0:model/group_normalization/strided_slice_1/stack_1:output:0:model/group_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
/model/group_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/group_normalization/strided_slice_2StridedSlice*model/group_normalization/Shape_1:output:08model/group_normalization/strided_slice_2/stack:output:0:model/group_normalization/strided_slice_2/stack_1:output:0:model/group_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
/model/group_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/group_normalization/strided_slice_3StridedSlice*model/group_normalization/Shape_1:output:08model/group_normalization/strided_slice_3/stack:output:0:model/group_normalization/strided_slice_3/stack_1:output:0:model/group_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
/model/group_normalization/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1model/group_normalization/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/group_normalization/strided_slice_4StridedSlice*model/group_normalization/Shape_1:output:08model/group_normalization/strided_slice_4/stack:output:0:model/group_normalization/strided_slice_4/stack_1:output:0:model/group_normalization/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$model/group_normalization/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :В
"model/group_normalization/floordivFloorDiv2model/group_normalization/strided_slice_4:output:0-model/group_normalization/floordiv/y:output:0*
T0*
_output_shapes
: c
!model/group_normalization/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
model/group_normalization/stackPack0model/group_normalization/strided_slice:output:02model/group_normalization/strided_slice_1:output:02model/group_normalization/strided_slice_2:output:0*model/group_normalization/stack/3:output:0&model/group_normalization/floordiv:z:0*
N*
T0*
_output_shapes
:а
!model/group_normalization/ReshapeReshape(model/average_pooling2d/AvgPool:output:0(model/group_normalization/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
8model/group_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ь
&model/group_normalization/moments/meanMean*model/group_normalization/Reshape:output:0Amodel/group_normalization/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(­
.model/group_normalization/moments/StopGradientStopGradient/model/group_normalization/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ§
3model/group_normalization/moments/SquaredDifferenceSquaredDifference*model/group_normalization/Reshape:output:07model/group_normalization/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<model/group_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
*model/group_normalization/moments/varianceMean7model/group_normalization/moments/SquaredDifference:z:0Emodel/group_normalization/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(
/model/group_normalization/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1model/group_normalization/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
)model/group_normalization/strided_slice_5StridedSlice(model/group_normalization/Shape:output:08model/group_normalization/strided_slice_5/stack:output:0:model/group_normalization/strided_slice_5/stack_1:output:0:model/group_normalization/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
$model/group_normalization/floordiv_1FloorDiv2model/group_normalization/strided_slice_5:output:0/model/group_normalization/floordiv_1/y:output:0*
T0*
_output_shapes
: Њ
2model/group_normalization/Reshape_1/ReadVariableOpReadVariableOp;model_group_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0m
+model/group_normalization/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/group_normalization/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :m
+model/group_normalization/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+model/group_normalization/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :с
)model/group_normalization/Reshape_1/shapePack4model/group_normalization/Reshape_1/shape/0:output:04model/group_normalization/Reshape_1/shape/1:output:04model/group_normalization/Reshape_1/shape/2:output:04model/group_normalization/Reshape_1/shape/3:output:0(model/group_normalization/floordiv_1:z:0*
N*
T0*
_output_shapes
:г
#model/group_normalization/Reshape_1Reshape:model/group_normalization/Reshape_1/ReadVariableOp:value:02model/group_normalization/Reshape_1/shape:output:0*
T0**
_output_shapes
:Њ
2model/group_normalization/Reshape_2/ReadVariableOpReadVariableOp;model_group_normalization_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0m
+model/group_normalization/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/group_normalization/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :m
+model/group_normalization/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+model/group_normalization/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :с
)model/group_normalization/Reshape_2/shapePack4model/group_normalization/Reshape_2/shape/0:output:04model/group_normalization/Reshape_2/shape/1:output:04model/group_normalization/Reshape_2/shape/2:output:04model/group_normalization/Reshape_2/shape/3:output:0(model/group_normalization/floordiv_1:z:0*
N*
T0*
_output_shapes
:г
#model/group_normalization/Reshape_2Reshape:model/group_normalization/Reshape_2/ReadVariableOp:value:02model/group_normalization/Reshape_2/shape:output:0*
T0**
_output_shapes
:n
)model/group_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:з
'model/group_normalization/batchnorm/addAddV23model/group_normalization/moments/variance:output:02model/group_normalization/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
)model/group_normalization/batchnorm/RsqrtRsqrt+model/group_normalization/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЩ
'model/group_normalization/batchnorm/mulMul-model/group_normalization/batchnorm/Rsqrt:y:0,model/group_normalization/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџй
)model/group_normalization/batchnorm/mul_1Mul*model/group_normalization/Reshape:output:0+model/group_normalization/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџЬ
)model/group_normalization/batchnorm/mul_2Mul/model/group_normalization/moments/mean:output:0+model/group_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЩ
'model/group_normalization/batchnorm/subSub,model/group_normalization/Reshape_2:output:0-model/group_normalization/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџо
)model/group_normalization/batchnorm/add_1AddV2-model/group_normalization/batchnorm/mul_1:z:0+model/group_normalization/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџг
#model/group_normalization/Reshape_3Reshape-model/group_normalization/batchnorm/add_1:z:0(model/group_normalization/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/re_lu/ReluRelu,model/group_normalization/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0с
model/conv2d_1/Conv2DConv2Dmodel/re_lu/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџе
!model/average_pooling2d_1/AvgPoolAvgPoolmodel/conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides

!model/group_normalization_1/ShapeShape*model/average_pooling2d_1/AvgPool:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_1/Shape_1Shape*model/average_pooling2d_1/AvgPool:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_1/strided_sliceStridedSlice,model/group_normalization_1/Shape_1:output:08model/group_normalization_1/strided_slice/stack:output:0:model/group_normalization_1/strided_slice/stack_1:output:0:model/group_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_1/strided_slice_1StridedSlice,model/group_normalization_1/Shape_1:output:0:model/group_normalization_1/strided_slice_1/stack:output:0<model/group_normalization_1/strided_slice_1/stack_1:output:0<model/group_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_1/strided_slice_2StridedSlice,model/group_normalization_1/Shape_1:output:0:model/group_normalization_1/strided_slice_2/stack:output:0<model/group_normalization_1/strided_slice_2/stack_1:output:0<model/group_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_1/strided_slice_3StridedSlice,model/group_normalization_1/Shape_1:output:0:model/group_normalization_1/strided_slice_3/stack:output:0<model/group_normalization_1/strided_slice_3/stack_1:output:0<model/group_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_1/strided_slice_4StridedSlice,model/group_normalization_1/Shape_1:output:0:model/group_normalization_1/strided_slice_4/stack:output:0<model/group_normalization_1/strided_slice_4/stack_1:output:0<model/group_normalization_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_1/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :И
$model/group_normalization_1/floordivFloorDiv4model/group_normalization_1/strided_slice_4:output:0/model/group_normalization_1/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Я
!model/group_normalization_1/stackPack2model/group_normalization_1/strided_slice:output:04model/group_normalization_1/strided_slice_1:output:04model/group_normalization_1/strided_slice_2:output:0,model/group_normalization_1/stack/3:output:0(model/group_normalization_1/floordiv:z:0*
N*
T0*
_output_shapes
:ж
#model/group_normalization_1/ReshapeReshape*model/average_pooling2d_1/AvgPool:output:0*model/group_normalization_1/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
:model/group_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_1/moments/meanMean,model/group_normalization_1/Reshape:output:0Cmodel/group_normalization_1/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(Б
0model/group_normalization_1/moments/StopGradientStopGradient1model/group_normalization_1/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
5model/group_normalization_1/moments/SquaredDifferenceSquaredDifference,model/group_normalization_1/Reshape:output:09model/group_normalization_1/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
>model/group_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_1/moments/varianceMean9model/group_normalization_1/moments/SquaredDifference:z:0Gmodel/group_normalization_1/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(
1model/group_normalization_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_1/strided_slice_5StridedSlice*model/group_normalization_1/Shape:output:0:model/group_normalization_1/strided_slice_5/stack:output:0<model/group_normalization_1/strided_slice_5/stack_1:output:0<model/group_normalization_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_1/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :М
&model/group_normalization_1/floordiv_1FloorDiv4model/group_normalization_1/strided_slice_5:output:01model/group_normalization_1/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_1/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_1_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_1/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_1/Reshape_1/shapePack6model/group_normalization_1/Reshape_1/shape/0:output:06model/group_normalization_1/Reshape_1/shape/1:output:06model/group_normalization_1/Reshape_1/shape/2:output:06model/group_normalization_1/Reshape_1/shape/3:output:0*model/group_normalization_1/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_1/Reshape_1Reshape<model/group_normalization_1/Reshape_1/ReadVariableOp:value:04model/group_normalization_1/Reshape_1/shape:output:0*
T0**
_output_shapes
:Ў
4model/group_normalization_1/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_1_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_1/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_1/Reshape_2/shapePack6model/group_normalization_1/Reshape_2/shape/0:output:06model/group_normalization_1/Reshape_2/shape/1:output:06model/group_normalization_1/Reshape_2/shape/2:output:06model/group_normalization_1/Reshape_2/shape/3:output:0*model/group_normalization_1/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_1/Reshape_2Reshape<model/group_normalization_1/Reshape_2/ReadVariableOp:value:04model/group_normalization_1/Reshape_2/shape:output:0*
T0**
_output_shapes
:p
+model/group_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_1/batchnorm/addAddV25model/group_normalization_1/moments/variance:output:04model/group_normalization_1/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЁ
+model/group_normalization_1/batchnorm/RsqrtRsqrt-model/group_normalization_1/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_1/batchnorm/mulMul/model/group_normalization_1/batchnorm/Rsqrt:y:0.model/group_normalization_1/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџп
+model/group_normalization_1/batchnorm/mul_1Mul,model/group_normalization_1/Reshape:output:0-model/group_normalization_1/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџв
+model/group_normalization_1/batchnorm/mul_2Mul1model/group_normalization_1/moments/mean:output:0-model/group_normalization_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_1/batchnorm/subSub.model/group_normalization_1/Reshape_2:output:0/model/group_normalization_1/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџф
+model/group_normalization_1/batchnorm/add_1AddV2/model/group_normalization_1/batchnorm/mul_1:z:0-model/group_normalization_1/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџй
%model/group_normalization_1/Reshape_3Reshape/model/group_normalization_1/batchnorm/add_1:z:0*model/group_normalization_1/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/re_lu_1/ReluRelu.model/group_normalization_1/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0у
model/conv2d_2/Conv2DConv2D model/re_lu_1/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџе
!model/average_pooling2d_2/AvgPoolAvgPoolmodel/conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides

!model/group_normalization_2/ShapeShape*model/average_pooling2d_2/AvgPool:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_2/Shape_1Shape*model/average_pooling2d_2/AvgPool:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_2/strided_sliceStridedSlice,model/group_normalization_2/Shape_1:output:08model/group_normalization_2/strided_slice/stack:output:0:model/group_normalization_2/strided_slice/stack_1:output:0:model/group_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_2/strided_slice_1StridedSlice,model/group_normalization_2/Shape_1:output:0:model/group_normalization_2/strided_slice_1/stack:output:0<model/group_normalization_2/strided_slice_1/stack_1:output:0<model/group_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_2/strided_slice_2StridedSlice,model/group_normalization_2/Shape_1:output:0:model/group_normalization_2/strided_slice_2/stack:output:0<model/group_normalization_2/strided_slice_2/stack_1:output:0<model/group_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_2/strided_slice_3StridedSlice,model/group_normalization_2/Shape_1:output:0:model/group_normalization_2/strided_slice_3/stack:output:0<model/group_normalization_2/strided_slice_3/stack_1:output:0<model/group_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_2/strided_slice_4StridedSlice,model/group_normalization_2/Shape_1:output:0:model/group_normalization_2/strided_slice_4/stack:output:0<model/group_normalization_2/strided_slice_4/stack_1:output:0<model/group_normalization_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_2/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :И
$model/group_normalization_2/floordivFloorDiv4model/group_normalization_2/strided_slice_4:output:0/model/group_normalization_2/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Я
!model/group_normalization_2/stackPack2model/group_normalization_2/strided_slice:output:04model/group_normalization_2/strided_slice_1:output:04model/group_normalization_2/strided_slice_2:output:0,model/group_normalization_2/stack/3:output:0(model/group_normalization_2/floordiv:z:0*
N*
T0*
_output_shapes
:ж
#model/group_normalization_2/ReshapeReshape*model/average_pooling2d_2/AvgPool:output:0*model/group_normalization_2/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
:model/group_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_2/moments/meanMean,model/group_normalization_2/Reshape:output:0Cmodel/group_normalization_2/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(Б
0model/group_normalization_2/moments/StopGradientStopGradient1model/group_normalization_2/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
5model/group_normalization_2/moments/SquaredDifferenceSquaredDifference,model/group_normalization_2/Reshape:output:09model/group_normalization_2/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
>model/group_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_2/moments/varianceMean9model/group_normalization_2/moments/SquaredDifference:z:0Gmodel/group_normalization_2/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(
1model/group_normalization_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_2/strided_slice_5StridedSlice*model/group_normalization_2/Shape:output:0:model/group_normalization_2/strided_slice_5/stack:output:0<model/group_normalization_2/strided_slice_5/stack_1:output:0<model/group_normalization_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_2/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :М
&model/group_normalization_2/floordiv_1FloorDiv4model/group_normalization_2/strided_slice_5:output:01model/group_normalization_2/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_2/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_2_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_2/Reshape_1/shapePack6model/group_normalization_2/Reshape_1/shape/0:output:06model/group_normalization_2/Reshape_1/shape/1:output:06model/group_normalization_2/Reshape_1/shape/2:output:06model/group_normalization_2/Reshape_1/shape/3:output:0*model/group_normalization_2/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_2/Reshape_1Reshape<model/group_normalization_2/Reshape_1/ReadVariableOp:value:04model/group_normalization_2/Reshape_1/shape:output:0*
T0**
_output_shapes
:Ў
4model/group_normalization_2/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_2_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_2/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_2/Reshape_2/shapePack6model/group_normalization_2/Reshape_2/shape/0:output:06model/group_normalization_2/Reshape_2/shape/1:output:06model/group_normalization_2/Reshape_2/shape/2:output:06model/group_normalization_2/Reshape_2/shape/3:output:0*model/group_normalization_2/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_2/Reshape_2Reshape<model/group_normalization_2/Reshape_2/ReadVariableOp:value:04model/group_normalization_2/Reshape_2/shape:output:0*
T0**
_output_shapes
:p
+model/group_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_2/batchnorm/addAddV25model/group_normalization_2/moments/variance:output:04model/group_normalization_2/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЁ
+model/group_normalization_2/batchnorm/RsqrtRsqrt-model/group_normalization_2/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_2/batchnorm/mulMul/model/group_normalization_2/batchnorm/Rsqrt:y:0.model/group_normalization_2/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџп
+model/group_normalization_2/batchnorm/mul_1Mul,model/group_normalization_2/Reshape:output:0-model/group_normalization_2/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџв
+model/group_normalization_2/batchnorm/mul_2Mul1model/group_normalization_2/moments/mean:output:0-model/group_normalization_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_2/batchnorm/subSub.model/group_normalization_2/Reshape_2:output:0/model/group_normalization_2/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџф
+model/group_normalization_2/batchnorm/add_1AddV2/model/group_normalization_2/batchnorm/mul_1:z:0-model/group_normalization_2/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџй
%model/group_normalization_2/Reshape_3Reshape/model/group_normalization_2/batchnorm/add_1:z:0*model/group_normalization_2/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/re_lu_2/ReluRelu.model/group_normalization_2/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0у
model/conv2d_3/Conv2DConv2D model/re_lu_2/Relu:activations:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides

%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0М
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ е
!model/average_pooling2d_3/AvgPoolAvgPoolmodel/conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
ksize
*
paddingVALID*
strides

!model/group_normalization_3/ShapeShape*model/average_pooling2d_3/AvgPool:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_3/Shape_1Shape*model/average_pooling2d_3/AvgPool:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_3/strided_sliceStridedSlice,model/group_normalization_3/Shape_1:output:08model/group_normalization_3/strided_slice/stack:output:0:model/group_normalization_3/strided_slice/stack_1:output:0:model/group_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_3/strided_slice_1StridedSlice,model/group_normalization_3/Shape_1:output:0:model/group_normalization_3/strided_slice_1/stack:output:0<model/group_normalization_3/strided_slice_1/stack_1:output:0<model/group_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_3/strided_slice_2StridedSlice,model/group_normalization_3/Shape_1:output:0:model/group_normalization_3/strided_slice_2/stack:output:0<model/group_normalization_3/strided_slice_2/stack_1:output:0<model/group_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_3/strided_slice_3StridedSlice,model/group_normalization_3/Shape_1:output:0:model/group_normalization_3/strided_slice_3/stack:output:0<model/group_normalization_3/strided_slice_3/stack_1:output:0<model/group_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_3/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_3/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_3/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_3/strided_slice_4StridedSlice,model/group_normalization_3/Shape_1:output:0:model/group_normalization_3/strided_slice_4/stack:output:0<model/group_normalization_3/strided_slice_4/stack_1:output:0<model/group_normalization_3/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_3/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B : И
$model/group_normalization_3/floordivFloorDiv4model/group_normalization_3/strided_slice_4:output:0/model/group_normalization_3/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Я
!model/group_normalization_3/stackPack2model/group_normalization_3/strided_slice:output:04model/group_normalization_3/strided_slice_1:output:04model/group_normalization_3/strided_slice_2:output:0,model/group_normalization_3/stack/3:output:0(model/group_normalization_3/floordiv:z:0*
N*
T0*
_output_shapes
:ж
#model/group_normalization_3/ReshapeReshape*model/average_pooling2d_3/AvgPool:output:0*model/group_normalization_3/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
:model/group_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_3/moments/meanMean,model/group_normalization_3/Reshape:output:0Cmodel/group_normalization_3/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(Б
0model/group_normalization_3/moments/StopGradientStopGradient1model/group_normalization_3/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
5model/group_normalization_3/moments/SquaredDifferenceSquaredDifference,model/group_normalization_3/Reshape:output:09model/group_normalization_3/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
>model/group_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_3/moments/varianceMean9model/group_normalization_3/moments/SquaredDifference:z:0Gmodel/group_normalization_3/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(
1model/group_normalization_3/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_3/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_3/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_3/strided_slice_5StridedSlice*model/group_normalization_3/Shape:output:0:model/group_normalization_3/strided_slice_5/stack:output:0<model/group_normalization_3/strided_slice_5/stack_1:output:0<model/group_normalization_3/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_3/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B : М
&model/group_normalization_3/floordiv_1FloorDiv4model/group_normalization_3/strided_slice_5:output:01model/group_normalization_3/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_3/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0o
-model/group_normalization_3/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
+model/group_normalization_3/Reshape_1/shapePack6model/group_normalization_3/Reshape_1/shape/0:output:06model/group_normalization_3/Reshape_1/shape/1:output:06model/group_normalization_3/Reshape_1/shape/2:output:06model/group_normalization_3/Reshape_1/shape/3:output:0*model/group_normalization_3/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_3/Reshape_1Reshape<model/group_normalization_3/Reshape_1/ReadVariableOp:value:04model/group_normalization_3/Reshape_1/shape:output:0*
T0**
_output_shapes
: Ў
4model/group_normalization_3/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_3_reshape_2_readvariableop_resource*
_output_shapes
: *
dtype0o
-model/group_normalization_3/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
+model/group_normalization_3/Reshape_2/shapePack6model/group_normalization_3/Reshape_2/shape/0:output:06model/group_normalization_3/Reshape_2/shape/1:output:06model/group_normalization_3/Reshape_2/shape/2:output:06model/group_normalization_3/Reshape_2/shape/3:output:0*model/group_normalization_3/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_3/Reshape_2Reshape<model/group_normalization_3/Reshape_2/ReadVariableOp:value:04model/group_normalization_3/Reshape_2/shape:output:0*
T0**
_output_shapes
: p
+model/group_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_3/batchnorm/addAddV25model/group_normalization_3/moments/variance:output:04model/group_normalization_3/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ Ё
+model/group_normalization_3/batchnorm/RsqrtRsqrt-model/group_normalization_3/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ Я
)model/group_normalization_3/batchnorm/mulMul/model/group_normalization_3/batchnorm/Rsqrt:y:0.model/group_normalization_3/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ п
+model/group_normalization_3/batchnorm/mul_1Mul,model/group_normalization_3/Reshape:output:0-model/group_normalization_3/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ в
+model/group_normalization_3/batchnorm/mul_2Mul1model/group_normalization_3/moments/mean:output:0-model/group_normalization_3/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ Я
)model/group_normalization_3/batchnorm/subSub.model/group_normalization_3/Reshape_2:output:0/model/group_normalization_3/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ ф
+model/group_normalization_3/batchnorm/add_1AddV2/model/group_normalization_3/batchnorm/mul_1:z:0-model/group_normalization_3/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
%model/group_normalization_3/Reshape_3Reshape/model/group_normalization_3/batchnorm/add_1:z:0*model/group_normalization_3/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
model/re_lu_3/ReluRelu.model/group_normalization_3/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0у
model/conv2d_4/Conv2DConv2D model/re_lu_3/Relu:activations:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides

%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@е
!model/average_pooling2d_4/AvgPoolAvgPoolmodel/conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
ksize
*
paddingVALID*
strides

!model/group_normalization_4/ShapeShape*model/average_pooling2d_4/AvgPool:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_4/Shape_1Shape*model/average_pooling2d_4/AvgPool:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_4/strided_sliceStridedSlice,model/group_normalization_4/Shape_1:output:08model/group_normalization_4/strided_slice/stack:output:0:model/group_normalization_4/strided_slice/stack_1:output:0:model/group_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_4/strided_slice_1StridedSlice,model/group_normalization_4/Shape_1:output:0:model/group_normalization_4/strided_slice_1/stack:output:0<model/group_normalization_4/strided_slice_1/stack_1:output:0<model/group_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_4/strided_slice_2StridedSlice,model/group_normalization_4/Shape_1:output:0:model/group_normalization_4/strided_slice_2/stack:output:0<model/group_normalization_4/strided_slice_2/stack_1:output:0<model/group_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_4/strided_slice_3StridedSlice,model/group_normalization_4/Shape_1:output:0:model/group_normalization_4/strided_slice_3/stack:output:0<model/group_normalization_4/strided_slice_3/stack_1:output:0<model/group_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_4/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_4/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_4/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_4/strided_slice_4StridedSlice,model/group_normalization_4/Shape_1:output:0:model/group_normalization_4/strided_slice_4/stack:output:0<model/group_normalization_4/strided_slice_4/stack_1:output:0<model/group_normalization_4/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_4/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :@И
$model/group_normalization_4/floordivFloorDiv4model/group_normalization_4/strided_slice_4:output:0/model/group_normalization_4/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Я
!model/group_normalization_4/stackPack2model/group_normalization_4/strided_slice:output:04model/group_normalization_4/strided_slice_1:output:04model/group_normalization_4/strided_slice_2:output:0,model/group_normalization_4/stack/3:output:0(model/group_normalization_4/floordiv:z:0*
N*
T0*
_output_shapes
:ж
#model/group_normalization_4/ReshapeReshape*model/average_pooling2d_4/AvgPool:output:0*model/group_normalization_4/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
:model/group_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_4/moments/meanMean,model/group_normalization_4/Reshape:output:0Cmodel/group_normalization_4/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@*
	keep_dims(Б
0model/group_normalization_4/moments/StopGradientStopGradient1model/group_normalization_4/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@
5model/group_normalization_4/moments/SquaredDifferenceSquaredDifference,model/group_normalization_4/Reshape:output:09model/group_normalization_4/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
>model/group_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_4/moments/varianceMean9model/group_normalization_4/moments/SquaredDifference:z:0Gmodel/group_normalization_4/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@*
	keep_dims(
1model/group_normalization_4/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_4/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_4/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_4/strided_slice_5StridedSlice*model/group_normalization_4/Shape:output:0:model/group_normalization_4/strided_slice_5/stack:output:0<model/group_normalization_4/strided_slice_5/stack_1:output:0<model/group_normalization_4/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_4/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :@М
&model/group_normalization_4/floordiv_1FloorDiv4model/group_normalization_4/strided_slice_5:output:01model/group_normalization_4/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_4/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0o
-model/group_normalization_4/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_4/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_4/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_4/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@э
+model/group_normalization_4/Reshape_1/shapePack6model/group_normalization_4/Reshape_1/shape/0:output:06model/group_normalization_4/Reshape_1/shape/1:output:06model/group_normalization_4/Reshape_1/shape/2:output:06model/group_normalization_4/Reshape_1/shape/3:output:0*model/group_normalization_4/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_4/Reshape_1Reshape<model/group_normalization_4/Reshape_1/ReadVariableOp:value:04model/group_normalization_4/Reshape_1/shape:output:0*
T0**
_output_shapes
:@Ў
4model/group_normalization_4/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_4_reshape_2_readvariableop_resource*
_output_shapes
:@*
dtype0o
-model/group_normalization_4/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_4/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_4/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_4/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@э
+model/group_normalization_4/Reshape_2/shapePack6model/group_normalization_4/Reshape_2/shape/0:output:06model/group_normalization_4/Reshape_2/shape/1:output:06model/group_normalization_4/Reshape_2/shape/2:output:06model/group_normalization_4/Reshape_2/shape/3:output:0*model/group_normalization_4/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_4/Reshape_2Reshape<model/group_normalization_4/Reshape_2/ReadVariableOp:value:04model/group_normalization_4/Reshape_2/shape:output:0*
T0**
_output_shapes
:@p
+model/group_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_4/batchnorm/addAddV25model/group_normalization_4/moments/variance:output:04model/group_normalization_4/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@Ё
+model/group_normalization_4/batchnorm/RsqrtRsqrt-model/group_normalization_4/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@Я
)model/group_normalization_4/batchnorm/mulMul/model/group_normalization_4/batchnorm/Rsqrt:y:0.model/group_normalization_4/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ@п
+model/group_normalization_4/batchnorm/mul_1Mul,model/group_normalization_4/Reshape:output:0-model/group_normalization_4/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@в
+model/group_normalization_4/batchnorm/mul_2Mul1model/group_normalization_4/moments/mean:output:0-model/group_normalization_4/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@Я
)model/group_normalization_4/batchnorm/subSub.model/group_normalization_4/Reshape_2:output:0/model/group_normalization_4/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ@ф
+model/group_normalization_4/batchnorm/add_1AddV2/model/group_normalization_4/batchnorm/mul_1:z:0-model/group_normalization_4/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ@й
%model/group_normalization_4/Reshape_3Reshape/model/group_normalization_4/batchnorm/add_1:z:0*model/group_normalization_4/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
model/re_lu_4/ReluRelu.model/group_normalization_4/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@w
model/up_sampling2d/ShapeShape model/re_lu_4/Relu:activations:0*
T0*
_output_shapes
::эЯq
'model/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)model/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)model/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
!model/up_sampling2d/strided_sliceStridedSlice"model/up_sampling2d/Shape:output:00model/up_sampling2d/strided_slice/stack:output:02model/up_sampling2d/strided_slice/stack_1:output:02model/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:j
model/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
model/up_sampling2d/mulMul*model/up_sampling2d/strided_slice:output:0"model/up_sampling2d/Const:output:0*
T0*
_output_shapes
:ю
0model/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor model/re_lu_4/Relu:activations:0model/up_sampling2d/mul:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
half_pixel_centers(
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0
model/conv2d_5/Conv2DConv2DAmodel/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides

%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0М
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ~
!model/group_normalization_5/ShapeShapemodel/conv2d_5/BiasAdd:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_5/Shape_1Shapemodel/conv2d_5/BiasAdd:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_5/strided_sliceStridedSlice,model/group_normalization_5/Shape_1:output:08model/group_normalization_5/strided_slice/stack:output:0:model/group_normalization_5/strided_slice/stack_1:output:0:model/group_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_5/strided_slice_1StridedSlice,model/group_normalization_5/Shape_1:output:0:model/group_normalization_5/strided_slice_1/stack:output:0<model/group_normalization_5/strided_slice_1/stack_1:output:0<model/group_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_5/strided_slice_2StridedSlice,model/group_normalization_5/Shape_1:output:0:model/group_normalization_5/strided_slice_2/stack:output:0<model/group_normalization_5/strided_slice_2/stack_1:output:0<model/group_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_5/strided_slice_3StridedSlice,model/group_normalization_5/Shape_1:output:0:model/group_normalization_5/strided_slice_3/stack:output:0<model/group_normalization_5/strided_slice_3/stack_1:output:0<model/group_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_5/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_5/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_5/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_5/strided_slice_4StridedSlice,model/group_normalization_5/Shape_1:output:0:model/group_normalization_5/strided_slice_4/stack:output:0<model/group_normalization_5/strided_slice_4/stack_1:output:0<model/group_normalization_5/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_5/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B : И
$model/group_normalization_5/floordivFloorDiv4model/group_normalization_5/strided_slice_4:output:0/model/group_normalization_5/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Я
!model/group_normalization_5/stackPack2model/group_normalization_5/strided_slice:output:04model/group_normalization_5/strided_slice_1:output:04model/group_normalization_5/strided_slice_2:output:0,model/group_normalization_5/stack/3:output:0(model/group_normalization_5/floordiv:z:0*
N*
T0*
_output_shapes
:Ы
#model/group_normalization_5/ReshapeReshapemodel/conv2d_5/BiasAdd:output:0*model/group_normalization_5/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
:model/group_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_5/moments/meanMean,model/group_normalization_5/Reshape:output:0Cmodel/group_normalization_5/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(Б
0model/group_normalization_5/moments/StopGradientStopGradient1model/group_normalization_5/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ 
5model/group_normalization_5/moments/SquaredDifferenceSquaredDifference,model/group_normalization_5/Reshape:output:09model/group_normalization_5/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
>model/group_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_5/moments/varianceMean9model/group_normalization_5/moments/SquaredDifference:z:0Gmodel/group_normalization_5/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ *
	keep_dims(
1model/group_normalization_5/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_5/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_5/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_5/strided_slice_5StridedSlice*model/group_normalization_5/Shape:output:0:model/group_normalization_5/strided_slice_5/stack:output:0<model/group_normalization_5/strided_slice_5/stack_1:output:0<model/group_normalization_5/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_5/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B : М
&model/group_normalization_5/floordiv_1FloorDiv4model/group_normalization_5/strided_slice_5:output:01model/group_normalization_5/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_5/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0o
-model/group_normalization_5/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_5/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_5/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_5/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
+model/group_normalization_5/Reshape_1/shapePack6model/group_normalization_5/Reshape_1/shape/0:output:06model/group_normalization_5/Reshape_1/shape/1:output:06model/group_normalization_5/Reshape_1/shape/2:output:06model/group_normalization_5/Reshape_1/shape/3:output:0*model/group_normalization_5/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_5/Reshape_1Reshape<model/group_normalization_5/Reshape_1/ReadVariableOp:value:04model/group_normalization_5/Reshape_1/shape:output:0*
T0**
_output_shapes
: Ў
4model/group_normalization_5/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_5_reshape_2_readvariableop_resource*
_output_shapes
: *
dtype0o
-model/group_normalization_5/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_5/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
+model/group_normalization_5/Reshape_2/shapePack6model/group_normalization_5/Reshape_2/shape/0:output:06model/group_normalization_5/Reshape_2/shape/1:output:06model/group_normalization_5/Reshape_2/shape/2:output:06model/group_normalization_5/Reshape_2/shape/3:output:0*model/group_normalization_5/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_5/Reshape_2Reshape<model/group_normalization_5/Reshape_2/ReadVariableOp:value:04model/group_normalization_5/Reshape_2/shape:output:0*
T0**
_output_shapes
: p
+model/group_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_5/batchnorm/addAddV25model/group_normalization_5/moments/variance:output:04model/group_normalization_5/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ Ё
+model/group_normalization_5/batchnorm/RsqrtRsqrt-model/group_normalization_5/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ Я
)model/group_normalization_5/batchnorm/mulMul/model/group_normalization_5/batchnorm/Rsqrt:y:0.model/group_normalization_5/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ п
+model/group_normalization_5/batchnorm/mul_1Mul,model/group_normalization_5/Reshape:output:0-model/group_normalization_5/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ в
+model/group_normalization_5/batchnorm/mul_2Mul1model/group_normalization_5/moments/mean:output:0-model/group_normalization_5/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ Я
)model/group_normalization_5/batchnorm/subSub.model/group_normalization_5/Reshape_2:output:0/model/group_normalization_5/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ ф
+model/group_normalization_5/batchnorm/add_1AddV2/model/group_normalization_5/batchnorm/mul_1:z:0-model/group_normalization_5/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
%model/group_normalization_5/Reshape_3Reshape/model/group_normalization_5/batchnorm/add_1:z:0*model/group_normalization_5/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
model/re_lu_5/ReluRelu.model/group_normalization_5/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
model/concatenate/concatConcatV2 model/re_lu_3/Relu:activations:0 model/re_lu_5/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@z
model/up_sampling2d_1/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
::эЯs
)model/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
#model/up_sampling2d_1/strided_sliceStridedSlice$model/up_sampling2d_1/Shape:output:02model/up_sampling2d_1/strided_slice/stack:output:04model/up_sampling2d_1/strided_slice/stack_1:output:04model/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
model/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
model/up_sampling2d_1/mulMul,model/up_sampling2d_1/strided_slice:output:0$model/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:ѓ
2model/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor!model/concatenate/concat:output:0model/up_sampling2d_1/mul:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
half_pixel_centers(
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
model/conv2d_6/Conv2DConv2DCmodel/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
!model/group_normalization_6/ShapeShapemodel/conv2d_6/BiasAdd:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_6/Shape_1Shapemodel/conv2d_6/BiasAdd:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_6/strided_sliceStridedSlice,model/group_normalization_6/Shape_1:output:08model/group_normalization_6/strided_slice/stack:output:0:model/group_normalization_6/strided_slice/stack_1:output:0:model/group_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_6/strided_slice_1StridedSlice,model/group_normalization_6/Shape_1:output:0:model/group_normalization_6/strided_slice_1/stack:output:0<model/group_normalization_6/strided_slice_1/stack_1:output:0<model/group_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_6/strided_slice_2StridedSlice,model/group_normalization_6/Shape_1:output:0:model/group_normalization_6/strided_slice_2/stack:output:0<model/group_normalization_6/strided_slice_2/stack_1:output:0<model/group_normalization_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_6/strided_slice_3StridedSlice,model/group_normalization_6/Shape_1:output:0:model/group_normalization_6/strided_slice_3/stack:output:0<model/group_normalization_6/strided_slice_3/stack_1:output:0<model/group_normalization_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_6/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_6/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_6/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_6/strided_slice_4StridedSlice,model/group_normalization_6/Shape_1:output:0:model/group_normalization_6/strided_slice_4/stack:output:0<model/group_normalization_6/strided_slice_4/stack_1:output:0<model/group_normalization_6/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_6/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :И
$model/group_normalization_6/floordivFloorDiv4model/group_normalization_6/strided_slice_4:output:0/model/group_normalization_6/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Я
!model/group_normalization_6/stackPack2model/group_normalization_6/strided_slice:output:04model/group_normalization_6/strided_slice_1:output:04model/group_normalization_6/strided_slice_2:output:0,model/group_normalization_6/stack/3:output:0(model/group_normalization_6/floordiv:z:0*
N*
T0*
_output_shapes
:Ы
#model/group_normalization_6/ReshapeReshapemodel/conv2d_6/BiasAdd:output:0*model/group_normalization_6/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
:model/group_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_6/moments/meanMean,model/group_normalization_6/Reshape:output:0Cmodel/group_normalization_6/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(Б
0model/group_normalization_6/moments/StopGradientStopGradient1model/group_normalization_6/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
5model/group_normalization_6/moments/SquaredDifferenceSquaredDifference,model/group_normalization_6/Reshape:output:09model/group_normalization_6/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
>model/group_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_6/moments/varianceMean9model/group_normalization_6/moments/SquaredDifference:z:0Gmodel/group_normalization_6/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(
1model/group_normalization_6/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_6/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_6/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_6/strided_slice_5StridedSlice*model/group_normalization_6/Shape:output:0:model/group_normalization_6/strided_slice_5/stack:output:0<model/group_normalization_6/strided_slice_5/stack_1:output:0<model/group_normalization_6/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_6/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :М
&model/group_normalization_6/floordiv_1FloorDiv4model/group_normalization_6/strided_slice_5:output:01model/group_normalization_6/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_6/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_6_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_6/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_6/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_6/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_6/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_6/Reshape_1/shapePack6model/group_normalization_6/Reshape_1/shape/0:output:06model/group_normalization_6/Reshape_1/shape/1:output:06model/group_normalization_6/Reshape_1/shape/2:output:06model/group_normalization_6/Reshape_1/shape/3:output:0*model/group_normalization_6/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_6/Reshape_1Reshape<model/group_normalization_6/Reshape_1/ReadVariableOp:value:04model/group_normalization_6/Reshape_1/shape:output:0*
T0**
_output_shapes
:Ў
4model/group_normalization_6/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_6_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_6/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_6/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_6/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_6/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_6/Reshape_2/shapePack6model/group_normalization_6/Reshape_2/shape/0:output:06model/group_normalization_6/Reshape_2/shape/1:output:06model/group_normalization_6/Reshape_2/shape/2:output:06model/group_normalization_6/Reshape_2/shape/3:output:0*model/group_normalization_6/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_6/Reshape_2Reshape<model/group_normalization_6/Reshape_2/ReadVariableOp:value:04model/group_normalization_6/Reshape_2/shape:output:0*
T0**
_output_shapes
:p
+model/group_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_6/batchnorm/addAddV25model/group_normalization_6/moments/variance:output:04model/group_normalization_6/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЁ
+model/group_normalization_6/batchnorm/RsqrtRsqrt-model/group_normalization_6/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_6/batchnorm/mulMul/model/group_normalization_6/batchnorm/Rsqrt:y:0.model/group_normalization_6/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџп
+model/group_normalization_6/batchnorm/mul_1Mul,model/group_normalization_6/Reshape:output:0-model/group_normalization_6/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџв
+model/group_normalization_6/batchnorm/mul_2Mul1model/group_normalization_6/moments/mean:output:0-model/group_normalization_6/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_6/batchnorm/subSub.model/group_normalization_6/Reshape_2:output:0/model/group_normalization_6/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџф
+model/group_normalization_6/batchnorm/add_1AddV2/model/group_normalization_6/batchnorm/mul_1:z:0-model/group_normalization_6/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџй
%model/group_normalization_6/Reshape_3Reshape/model/group_normalization_6/batchnorm/add_1:z:0*model/group_normalization_6/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/re_lu_6/ReluRelu.model/group_normalization_6/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :щ
model/concatenate_1/concatConcatV2 model/re_lu_2/Relu:activations:0 model/re_lu_6/Relu:activations:0(model/concatenate_1/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ |
model/up_sampling2d_2/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
::эЯs
)model/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
#model/up_sampling2d_2/strided_sliceStridedSlice$model/up_sampling2d_2/Shape:output:02model/up_sampling2d_2/strided_slice/stack:output:04model/up_sampling2d_2/strided_slice/stack_1:output:04model/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
model/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
model/up_sampling2d_2/mulMul,model/up_sampling2d_2/strided_slice:output:0$model/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:ѕ
2model/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor#model/concatenate_1/concat:output:0model/up_sampling2d_2/mul:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
half_pixel_centers(
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
model/conv2d_7/Conv2DConv2DCmodel/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
!model/group_normalization_7/ShapeShapemodel/conv2d_7/BiasAdd:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_7/Shape_1Shapemodel/conv2d_7/BiasAdd:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_7/strided_sliceStridedSlice,model/group_normalization_7/Shape_1:output:08model/group_normalization_7/strided_slice/stack:output:0:model/group_normalization_7/strided_slice/stack_1:output:0:model/group_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_7/strided_slice_1StridedSlice,model/group_normalization_7/Shape_1:output:0:model/group_normalization_7/strided_slice_1/stack:output:0<model/group_normalization_7/strided_slice_1/stack_1:output:0<model/group_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_7/strided_slice_2StridedSlice,model/group_normalization_7/Shape_1:output:0:model/group_normalization_7/strided_slice_2/stack:output:0<model/group_normalization_7/strided_slice_2/stack_1:output:0<model/group_normalization_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_7/strided_slice_3StridedSlice,model/group_normalization_7/Shape_1:output:0:model/group_normalization_7/strided_slice_3/stack:output:0<model/group_normalization_7/strided_slice_3/stack_1:output:0<model/group_normalization_7/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_7/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_7/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_7/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_7/strided_slice_4StridedSlice,model/group_normalization_7/Shape_1:output:0:model/group_normalization_7/strided_slice_4/stack:output:0<model/group_normalization_7/strided_slice_4/stack_1:output:0<model/group_normalization_7/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_7/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :И
$model/group_normalization_7/floordivFloorDiv4model/group_normalization_7/strided_slice_4:output:0/model/group_normalization_7/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Я
!model/group_normalization_7/stackPack2model/group_normalization_7/strided_slice:output:04model/group_normalization_7/strided_slice_1:output:04model/group_normalization_7/strided_slice_2:output:0,model/group_normalization_7/stack/3:output:0(model/group_normalization_7/floordiv:z:0*
N*
T0*
_output_shapes
:Ы
#model/group_normalization_7/ReshapeReshapemodel/conv2d_7/BiasAdd:output:0*model/group_normalization_7/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
:model/group_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_7/moments/meanMean,model/group_normalization_7/Reshape:output:0Cmodel/group_normalization_7/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(Б
0model/group_normalization_7/moments/StopGradientStopGradient1model/group_normalization_7/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
5model/group_normalization_7/moments/SquaredDifferenceSquaredDifference,model/group_normalization_7/Reshape:output:09model/group_normalization_7/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
>model/group_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_7/moments/varianceMean9model/group_normalization_7/moments/SquaredDifference:z:0Gmodel/group_normalization_7/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(
1model/group_normalization_7/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_7/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_7/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_7/strided_slice_5StridedSlice*model/group_normalization_7/Shape:output:0:model/group_normalization_7/strided_slice_5/stack:output:0<model/group_normalization_7/strided_slice_5/stack_1:output:0<model/group_normalization_7/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_7/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :М
&model/group_normalization_7/floordiv_1FloorDiv4model/group_normalization_7/strided_slice_5:output:01model/group_normalization_7/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_7/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_7_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_7/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_7/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_7/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_7/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_7/Reshape_1/shapePack6model/group_normalization_7/Reshape_1/shape/0:output:06model/group_normalization_7/Reshape_1/shape/1:output:06model/group_normalization_7/Reshape_1/shape/2:output:06model/group_normalization_7/Reshape_1/shape/3:output:0*model/group_normalization_7/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_7/Reshape_1Reshape<model/group_normalization_7/Reshape_1/ReadVariableOp:value:04model/group_normalization_7/Reshape_1/shape:output:0*
T0**
_output_shapes
:Ў
4model/group_normalization_7/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_7_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_7/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_7/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_7/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_7/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_7/Reshape_2/shapePack6model/group_normalization_7/Reshape_2/shape/0:output:06model/group_normalization_7/Reshape_2/shape/1:output:06model/group_normalization_7/Reshape_2/shape/2:output:06model/group_normalization_7/Reshape_2/shape/3:output:0*model/group_normalization_7/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_7/Reshape_2Reshape<model/group_normalization_7/Reshape_2/ReadVariableOp:value:04model/group_normalization_7/Reshape_2/shape:output:0*
T0**
_output_shapes
:p
+model/group_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_7/batchnorm/addAddV25model/group_normalization_7/moments/variance:output:04model/group_normalization_7/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЁ
+model/group_normalization_7/batchnorm/RsqrtRsqrt-model/group_normalization_7/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_7/batchnorm/mulMul/model/group_normalization_7/batchnorm/Rsqrt:y:0.model/group_normalization_7/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџп
+model/group_normalization_7/batchnorm/mul_1Mul,model/group_normalization_7/Reshape:output:0-model/group_normalization_7/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџв
+model/group_normalization_7/batchnorm/mul_2Mul1model/group_normalization_7/moments/mean:output:0-model/group_normalization_7/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_7/batchnorm/subSub.model/group_normalization_7/Reshape_2:output:0/model/group_normalization_7/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџф
+model/group_normalization_7/batchnorm/add_1AddV2/model/group_normalization_7/batchnorm/mul_1:z:0-model/group_normalization_7/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџй
%model/group_normalization_7/Reshape_3Reshape/model/group_normalization_7/batchnorm/add_1:z:0*model/group_normalization_7/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/re_lu_7/ReluRelu.model/group_normalization_7/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :щ
model/concatenate_2/concatConcatV2 model/re_lu_1/Relu:activations:0 model/re_lu_7/Relu:activations:0(model/concatenate_2/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
model/up_sampling2d_3/ShapeShape#model/concatenate_2/concat:output:0*
T0*
_output_shapes
::эЯs
)model/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
#model/up_sampling2d_3/strided_sliceStridedSlice$model/up_sampling2d_3/Shape:output:02model/up_sampling2d_3/strided_slice/stack:output:04model/up_sampling2d_3/strided_slice/stack_1:output:04model/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
model/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
model/up_sampling2d_3/mulMul,model/up_sampling2d_3/strided_slice:output:0$model/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:ѕ
2model/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor#model/concatenate_2/concat:output:0model/up_sampling2d_3/mul:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
$model/conv2d_8/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
model/conv2d_8/Conv2DConv2DCmodel/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

%model/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
model/conv2d_8/BiasAddBiasAddmodel/conv2d_8/Conv2D:output:0-model/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
!model/group_normalization_8/ShapeShapemodel/conv2d_8/BiasAdd:output:0*
T0*
_output_shapes
::эЯ
#model/group_normalization_8/Shape_1Shapemodel/conv2d_8/BiasAdd:output:0*
T0*
_output_shapes
::эЯy
/model/group_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/group_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/group_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
)model/group_normalization_8/strided_sliceStridedSlice,model/group_normalization_8/Shape_1:output:08model/group_normalization_8/strided_slice/stack:output:0:model/group_normalization_8/strided_slice/stack_1:output:0:model/group_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_8/strided_slice_1StridedSlice,model/group_normalization_8/Shape_1:output:0:model/group_normalization_8/strided_slice_1/stack:output:0<model/group_normalization_8/strided_slice_1/stack_1:output:0<model/group_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_8/strided_slice_2StridedSlice,model/group_normalization_8/Shape_1:output:0:model/group_normalization_8/strided_slice_2/stack:output:0<model/group_normalization_8/strided_slice_2/stack_1:output:0<model/group_normalization_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/group_normalization_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/group_normalization_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_8/strided_slice_3StridedSlice,model/group_normalization_8/Shape_1:output:0:model/group_normalization_8/strided_slice_3/stack:output:0<model/group_normalization_8/strided_slice_3/stack_1:output:0<model/group_normalization_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1model/group_normalization_8/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_8/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_8/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+model/group_normalization_8/strided_slice_4StridedSlice,model/group_normalization_8/Shape_1:output:0:model/group_normalization_8/strided_slice_4/stack:output:0<model/group_normalization_8/strided_slice_4/stack_1:output:0<model/group_normalization_8/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/group_normalization_8/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :И
$model/group_normalization_8/floordivFloorDiv4model/group_normalization_8/strided_slice_4:output:0/model/group_normalization_8/floordiv/y:output:0*
T0*
_output_shapes
: e
#model/group_normalization_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Я
!model/group_normalization_8/stackPack2model/group_normalization_8/strided_slice:output:04model/group_normalization_8/strided_slice_1:output:04model/group_normalization_8/strided_slice_2:output:0,model/group_normalization_8/stack/3:output:0(model/group_normalization_8/floordiv:z:0*
N*
T0*
_output_shapes
:Ы
#model/group_normalization_8/ReshapeReshapemodel/conv2d_8/BiasAdd:output:0*model/group_normalization_8/stack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
:model/group_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ђ
(model/group_normalization_8/moments/meanMean,model/group_normalization_8/Reshape:output:0Cmodel/group_normalization_8/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(Б
0model/group_normalization_8/moments/StopGradientStopGradient1model/group_normalization_8/moments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
5model/group_normalization_8/moments/SquaredDifferenceSquaredDifference,model/group_normalization_8/Reshape:output:09model/group_normalization_8/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
>model/group_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
,model/group_normalization_8/moments/varianceMean9model/group_normalization_8/moments/SquaredDifference:z:0Gmodel/group_normalization_8/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(
1model/group_normalization_8/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3model/group_normalization_8/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/group_normalization_8/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/group_normalization_8/strided_slice_5StridedSlice*model/group_normalization_8/Shape:output:0:model/group_normalization_8/strided_slice_5/stack:output:0<model/group_normalization_8/strided_slice_5/stack_1:output:0<model/group_normalization_8/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/group_normalization_8/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :М
&model/group_normalization_8/floordiv_1FloorDiv4model/group_normalization_8/strided_slice_5:output:01model/group_normalization_8/floordiv_1/y:output:0*
T0*
_output_shapes
: Ў
4model/group_normalization_8/Reshape_1/ReadVariableOpReadVariableOp=model_group_normalization_8_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_8/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_8/Reshape_1/shapePack6model/group_normalization_8/Reshape_1/shape/0:output:06model/group_normalization_8/Reshape_1/shape/1:output:06model/group_normalization_8/Reshape_1/shape/2:output:06model/group_normalization_8/Reshape_1/shape/3:output:0*model/group_normalization_8/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_8/Reshape_1Reshape<model/group_normalization_8/Reshape_1/ReadVariableOp:value:04model/group_normalization_8/Reshape_1/shape:output:0*
T0**
_output_shapes
:Ў
4model/group_normalization_8/Reshape_2/ReadVariableOpReadVariableOp=model_group_normalization_8_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0o
-model/group_normalization_8/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-model/group_normalization_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :э
+model/group_normalization_8/Reshape_2/shapePack6model/group_normalization_8/Reshape_2/shape/0:output:06model/group_normalization_8/Reshape_2/shape/1:output:06model/group_normalization_8/Reshape_2/shape/2:output:06model/group_normalization_8/Reshape_2/shape/3:output:0*model/group_normalization_8/floordiv_1:z:0*
N*
T0*
_output_shapes
:й
%model/group_normalization_8/Reshape_2Reshape<model/group_normalization_8/Reshape_2/ReadVariableOp:value:04model/group_normalization_8/Reshape_2/shape:output:0*
T0**
_output_shapes
:p
+model/group_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
)model/group_normalization_8/batchnorm/addAddV25model/group_normalization_8/moments/variance:output:04model/group_normalization_8/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЁ
+model/group_normalization_8/batchnorm/RsqrtRsqrt-model/group_normalization_8/batchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_8/batchnorm/mulMul/model/group_normalization_8/batchnorm/Rsqrt:y:0.model/group_normalization_8/Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџп
+model/group_normalization_8/batchnorm/mul_1Mul,model/group_normalization_8/Reshape:output:0-model/group_normalization_8/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџв
+model/group_normalization_8/batchnorm/mul_2Mul1model/group_normalization_8/moments/mean:output:0-model/group_normalization_8/batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџЯ
)model/group_normalization_8/batchnorm/subSub.model/group_normalization_8/Reshape_2:output:0/model/group_normalization_8/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџф
+model/group_normalization_8/batchnorm/add_1AddV2/model/group_normalization_8/batchnorm/mul_1:z:0-model/group_normalization_8/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџй
%model/group_normalization_8/Reshape_3Reshape/model/group_normalization_8/batchnorm/add_1:z:0*model/group_normalization_8/Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/re_lu_8/ReluRelu.model/group_normalization_8/Reshape_3:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч
model/concatenate_3/concatConcatV2model/re_lu/Relu:activations:0 model/re_lu_8/Relu:activations:0(model/concatenate_3/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
model/up_sampling2d_4/ShapeShape#model/concatenate_3/concat:output:0*
T0*
_output_shapes
::эЯs
)model/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+model/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
#model/up_sampling2d_4/strided_sliceStridedSlice$model/up_sampling2d_4/Shape:output:02model/up_sampling2d_4/strided_slice/stack:output:04model/up_sampling2d_4/strided_slice/stack_1:output:04model/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
model/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
model/up_sampling2d_4/mulMul,model/up_sampling2d_4/strided_slice:output:0$model/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:ѕ
2model/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor#model/concatenate_3/concat:output:0model/up_sampling2d_4/mul:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
$model/conv2d_9/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
model/conv2d_9/Conv2DConv2DCmodel/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides

%model/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
model/conv2d_9/BiasAddBiasAddmodel/conv2d_9/Conv2D:output:0-model/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentitymodel/conv2d_9/BiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp&^model/conv2d_7/BiasAdd/ReadVariableOp%^model/conv2d_7/Conv2D/ReadVariableOp&^model/conv2d_8/BiasAdd/ReadVariableOp%^model/conv2d_8/Conv2D/ReadVariableOp&^model/conv2d_9/BiasAdd/ReadVariableOp%^model/conv2d_9/Conv2D/ReadVariableOp3^model/group_normalization/Reshape_1/ReadVariableOp3^model/group_normalization/Reshape_2/ReadVariableOp5^model/group_normalization_1/Reshape_1/ReadVariableOp5^model/group_normalization_1/Reshape_2/ReadVariableOp5^model/group_normalization_2/Reshape_1/ReadVariableOp5^model/group_normalization_2/Reshape_2/ReadVariableOp5^model/group_normalization_3/Reshape_1/ReadVariableOp5^model/group_normalization_3/Reshape_2/ReadVariableOp5^model/group_normalization_4/Reshape_1/ReadVariableOp5^model/group_normalization_4/Reshape_2/ReadVariableOp5^model/group_normalization_5/Reshape_1/ReadVariableOp5^model/group_normalization_5/Reshape_2/ReadVariableOp5^model/group_normalization_6/Reshape_1/ReadVariableOp5^model/group_normalization_6/Reshape_2/ReadVariableOp5^model/group_normalization_7/Reshape_1/ReadVariableOp5^model/group_normalization_7/Reshape_2/ReadVariableOp5^model/group_normalization_8/Reshape_1/ReadVariableOp5^model/group_normalization_8/Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_7/BiasAdd/ReadVariableOp%model/conv2d_7/BiasAdd/ReadVariableOp2L
$model/conv2d_7/Conv2D/ReadVariableOp$model/conv2d_7/Conv2D/ReadVariableOp2N
%model/conv2d_8/BiasAdd/ReadVariableOp%model/conv2d_8/BiasAdd/ReadVariableOp2L
$model/conv2d_8/Conv2D/ReadVariableOp$model/conv2d_8/Conv2D/ReadVariableOp2N
%model/conv2d_9/BiasAdd/ReadVariableOp%model/conv2d_9/BiasAdd/ReadVariableOp2L
$model/conv2d_9/Conv2D/ReadVariableOp$model/conv2d_9/Conv2D/ReadVariableOp2h
2model/group_normalization/Reshape_1/ReadVariableOp2model/group_normalization/Reshape_1/ReadVariableOp2h
2model/group_normalization/Reshape_2/ReadVariableOp2model/group_normalization/Reshape_2/ReadVariableOp2l
4model/group_normalization_1/Reshape_1/ReadVariableOp4model/group_normalization_1/Reshape_1/ReadVariableOp2l
4model/group_normalization_1/Reshape_2/ReadVariableOp4model/group_normalization_1/Reshape_2/ReadVariableOp2l
4model/group_normalization_2/Reshape_1/ReadVariableOp4model/group_normalization_2/Reshape_1/ReadVariableOp2l
4model/group_normalization_2/Reshape_2/ReadVariableOp4model/group_normalization_2/Reshape_2/ReadVariableOp2l
4model/group_normalization_3/Reshape_1/ReadVariableOp4model/group_normalization_3/Reshape_1/ReadVariableOp2l
4model/group_normalization_3/Reshape_2/ReadVariableOp4model/group_normalization_3/Reshape_2/ReadVariableOp2l
4model/group_normalization_4/Reshape_1/ReadVariableOp4model/group_normalization_4/Reshape_1/ReadVariableOp2l
4model/group_normalization_4/Reshape_2/ReadVariableOp4model/group_normalization_4/Reshape_2/ReadVariableOp2l
4model/group_normalization_5/Reshape_1/ReadVariableOp4model/group_normalization_5/Reshape_1/ReadVariableOp2l
4model/group_normalization_5/Reshape_2/ReadVariableOp4model/group_normalization_5/Reshape_2/ReadVariableOp2l
4model/group_normalization_6/Reshape_1/ReadVariableOp4model/group_normalization_6/Reshape_1/ReadVariableOp2l
4model/group_normalization_6/Reshape_2/ReadVariableOp4model/group_normalization_6/Reshape_2/ReadVariableOp2l
4model/group_normalization_7/Reshape_1/ReadVariableOp4model/group_normalization_7/Reshape_1/ReadVariableOp2l
4model/group_normalization_7/Reshape_2/ReadVariableOp4model/group_normalization_7/Reshape_2/ReadVariableOp2l
4model/group_normalization_8/Reshape_1/ReadVariableOp4model/group_normalization_8/Reshape_1/ReadVariableOp2l
4model/group_normalization_8/Reshape_2/ReadVariableOp4model/group_normalization_8/Reshape_2/ReadVariableOp:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource
Г
ў
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6466743

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource

C
'__inference_re_lu_layer_call_fn_6467493

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_6466588z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_44391
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6466273

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6467604

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Я
V
"__inference__update_step_xla_44491
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	
 
7__inference_group_normalization_1_layer_call_fn_6467536

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6465897
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467530:'#
!
_user_specified_name	6467532
	

*__inference_conv2d_4_layer_call_fn_6467855

inputs!
unknown: @
	unknown_0:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6466680
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467849:'#
!
_user_specified_name	6467851
А
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6467730

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44506
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	

5__inference_group_normalization_layer_call_fn_6467420

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_group_normalization_layer_call_and_return_conditional_losses_6465810
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467414:'#
!
_user_specified_name	6467416
Ћ
J
"__inference__update_step_xla_44536
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_44556
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_44376
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
з
M
1__inference_up_sampling2d_3_layer_call_fn_6468375

inputs
identityї
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6466461
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
и
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6466836

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6468251

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6466346

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ
J
"__inference__update_step_xla_44486
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
т 
І	
%__inference_signature_wrapper_6467382
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: $

unknown_15: @

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@

unknown_24:

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_6465732
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1:'#
!
_user_specified_name	6467304:'#
!
_user_specified_name	6467306:'#
!
_user_specified_name	6467308:'#
!
_user_specified_name	6467310:'#
!
_user_specified_name	6467312:'#
!
_user_specified_name	6467314:'#
!
_user_specified_name	6467316:'#
!
_user_specified_name	6467318:'	#
!
_user_specified_name	6467320:'
#
!
_user_specified_name	6467322:'#
!
_user_specified_name	6467324:'#
!
_user_specified_name	6467326:'#
!
_user_specified_name	6467328:'#
!
_user_specified_name	6467330:'#
!
_user_specified_name	6467332:'#
!
_user_specified_name	6467334:'#
!
_user_specified_name	6467336:'#
!
_user_specified_name	6467338:'#
!
_user_specified_name	6467340:'#
!
_user_specified_name	6467342:'#
!
_user_specified_name	6467344:'#
!
_user_specified_name	6467346:'#
!
_user_specified_name	6467348:'#
!
_user_specified_name	6467350:'#
!
_user_specified_name	6467352:'#
!
_user_specified_name	6467354:'#
!
_user_specified_name	6467356:'#
!
_user_specified_name	6467358:'#
!
_user_specified_name	6467360:'#
!
_user_specified_name	6467362:'#
!
_user_specified_name	6467364:' #
!
_user_specified_name	6467366:'!#
!
_user_specified_name	6467368:'"#
!
_user_specified_name	6467370:'##
!
_user_specified_name	6467372:'$#
!
_user_specified_name	6467374:'%#
!
_user_specified_name	6467376:'&#
!
_user_specified_name	6467378
Г
ў
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6466813

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ў
^
B__inference_re_lu_layer_call_and_return_conditional_losses_6467498

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Q
5__inference_average_pooling2d_3_layer_call_fn_6467754

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6465998
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44516
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Я
V
"__inference__update_step_xla_44451
gradient"
variable: @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: @: *
	_noinline(:P L
&
_output_shapes
: @
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ё
l
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6467875

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44421
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6467411

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Б
ќ
C__inference_conv2d_layer_call_and_return_conditional_losses_6466572

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
І
q
 __inference__traced_save_6469272
file_prefix>
$read_disablecopyonread_conv2d_kernel:2
$read_1_disablecopyonread_conv2d_bias:@
2read_2_disablecopyonread_group_normalization_gamma:?
1read_3_disablecopyonread_group_normalization_beta:B
(read_4_disablecopyonread_conv2d_1_kernel:4
&read_5_disablecopyonread_conv2d_1_bias:B
4read_6_disablecopyonread_group_normalization_1_gamma:A
3read_7_disablecopyonread_group_normalization_1_beta:B
(read_8_disablecopyonread_conv2d_2_kernel:4
&read_9_disablecopyonread_conv2d_2_bias:C
5read_10_disablecopyonread_group_normalization_2_gamma:B
4read_11_disablecopyonread_group_normalization_2_beta:C
)read_12_disablecopyonread_conv2d_3_kernel: 5
'read_13_disablecopyonread_conv2d_3_bias: C
5read_14_disablecopyonread_group_normalization_3_gamma: B
4read_15_disablecopyonread_group_normalization_3_beta: C
)read_16_disablecopyonread_conv2d_4_kernel: @5
'read_17_disablecopyonread_conv2d_4_bias:@C
5read_18_disablecopyonread_group_normalization_4_gamma:@B
4read_19_disablecopyonread_group_normalization_4_beta:@C
)read_20_disablecopyonread_conv2d_5_kernel:@ 5
'read_21_disablecopyonread_conv2d_5_bias: C
5read_22_disablecopyonread_group_normalization_5_gamma: B
4read_23_disablecopyonread_group_normalization_5_beta: C
)read_24_disablecopyonread_conv2d_6_kernel:@5
'read_25_disablecopyonread_conv2d_6_bias:C
5read_26_disablecopyonread_group_normalization_6_gamma:B
4read_27_disablecopyonread_group_normalization_6_beta:C
)read_28_disablecopyonread_conv2d_7_kernel: 5
'read_29_disablecopyonread_conv2d_7_bias:C
5read_30_disablecopyonread_group_normalization_7_gamma:B
4read_31_disablecopyonread_group_normalization_7_beta:C
)read_32_disablecopyonread_conv2d_8_kernel:5
'read_33_disablecopyonread_conv2d_8_bias:C
5read_34_disablecopyonread_group_normalization_8_gamma:B
4read_35_disablecopyonread_group_normalization_8_beta:C
)read_36_disablecopyonread_conv2d_9_kernel:5
'read_37_disablecopyonread_conv2d_9_bias:-
#read_38_disablecopyonread_iteration:	 1
'read_39_disablecopyonread_learning_rate: H
.read_40_disablecopyonread_adam_m_conv2d_kernel:H
.read_41_disablecopyonread_adam_v_conv2d_kernel::
,read_42_disablecopyonread_adam_m_conv2d_bias::
,read_43_disablecopyonread_adam_v_conv2d_bias:H
:read_44_disablecopyonread_adam_m_group_normalization_gamma:H
:read_45_disablecopyonread_adam_v_group_normalization_gamma:G
9read_46_disablecopyonread_adam_m_group_normalization_beta:G
9read_47_disablecopyonread_adam_v_group_normalization_beta:J
0read_48_disablecopyonread_adam_m_conv2d_1_kernel:J
0read_49_disablecopyonread_adam_v_conv2d_1_kernel:<
.read_50_disablecopyonread_adam_m_conv2d_1_bias:<
.read_51_disablecopyonread_adam_v_conv2d_1_bias:J
<read_52_disablecopyonread_adam_m_group_normalization_1_gamma:J
<read_53_disablecopyonread_adam_v_group_normalization_1_gamma:I
;read_54_disablecopyonread_adam_m_group_normalization_1_beta:I
;read_55_disablecopyonread_adam_v_group_normalization_1_beta:J
0read_56_disablecopyonread_adam_m_conv2d_2_kernel:J
0read_57_disablecopyonread_adam_v_conv2d_2_kernel:<
.read_58_disablecopyonread_adam_m_conv2d_2_bias:<
.read_59_disablecopyonread_adam_v_conv2d_2_bias:J
<read_60_disablecopyonread_adam_m_group_normalization_2_gamma:J
<read_61_disablecopyonread_adam_v_group_normalization_2_gamma:I
;read_62_disablecopyonread_adam_m_group_normalization_2_beta:I
;read_63_disablecopyonread_adam_v_group_normalization_2_beta:J
0read_64_disablecopyonread_adam_m_conv2d_3_kernel: J
0read_65_disablecopyonread_adam_v_conv2d_3_kernel: <
.read_66_disablecopyonread_adam_m_conv2d_3_bias: <
.read_67_disablecopyonread_adam_v_conv2d_3_bias: J
<read_68_disablecopyonread_adam_m_group_normalization_3_gamma: J
<read_69_disablecopyonread_adam_v_group_normalization_3_gamma: I
;read_70_disablecopyonread_adam_m_group_normalization_3_beta: I
;read_71_disablecopyonread_adam_v_group_normalization_3_beta: J
0read_72_disablecopyonread_adam_m_conv2d_4_kernel: @J
0read_73_disablecopyonread_adam_v_conv2d_4_kernel: @<
.read_74_disablecopyonread_adam_m_conv2d_4_bias:@<
.read_75_disablecopyonread_adam_v_conv2d_4_bias:@J
<read_76_disablecopyonread_adam_m_group_normalization_4_gamma:@J
<read_77_disablecopyonread_adam_v_group_normalization_4_gamma:@I
;read_78_disablecopyonread_adam_m_group_normalization_4_beta:@I
;read_79_disablecopyonread_adam_v_group_normalization_4_beta:@J
0read_80_disablecopyonread_adam_m_conv2d_5_kernel:@ J
0read_81_disablecopyonread_adam_v_conv2d_5_kernel:@ <
.read_82_disablecopyonread_adam_m_conv2d_5_bias: <
.read_83_disablecopyonread_adam_v_conv2d_5_bias: J
<read_84_disablecopyonread_adam_m_group_normalization_5_gamma: J
<read_85_disablecopyonread_adam_v_group_normalization_5_gamma: I
;read_86_disablecopyonread_adam_m_group_normalization_5_beta: I
;read_87_disablecopyonread_adam_v_group_normalization_5_beta: J
0read_88_disablecopyonread_adam_m_conv2d_6_kernel:@J
0read_89_disablecopyonread_adam_v_conv2d_6_kernel:@<
.read_90_disablecopyonread_adam_m_conv2d_6_bias:<
.read_91_disablecopyonread_adam_v_conv2d_6_bias:J
<read_92_disablecopyonread_adam_m_group_normalization_6_gamma:J
<read_93_disablecopyonread_adam_v_group_normalization_6_gamma:I
;read_94_disablecopyonread_adam_m_group_normalization_6_beta:I
;read_95_disablecopyonread_adam_v_group_normalization_6_beta:J
0read_96_disablecopyonread_adam_m_conv2d_7_kernel: J
0read_97_disablecopyonread_adam_v_conv2d_7_kernel: <
.read_98_disablecopyonread_adam_m_conv2d_7_bias:<
.read_99_disablecopyonread_adam_v_conv2d_7_bias:K
=read_100_disablecopyonread_adam_m_group_normalization_7_gamma:K
=read_101_disablecopyonread_adam_v_group_normalization_7_gamma:J
<read_102_disablecopyonread_adam_m_group_normalization_7_beta:J
<read_103_disablecopyonread_adam_v_group_normalization_7_beta:K
1read_104_disablecopyonread_adam_m_conv2d_8_kernel:K
1read_105_disablecopyonread_adam_v_conv2d_8_kernel:=
/read_106_disablecopyonread_adam_m_conv2d_8_bias:=
/read_107_disablecopyonread_adam_v_conv2d_8_bias:K
=read_108_disablecopyonread_adam_m_group_normalization_8_gamma:K
=read_109_disablecopyonread_adam_v_group_normalization_8_gamma:J
<read_110_disablecopyonread_adam_m_group_normalization_8_beta:J
<read_111_disablecopyonread_adam_v_group_normalization_8_beta:K
1read_112_disablecopyonread_adam_m_conv2d_9_kernel:K
1read_113_disablecopyonread_adam_v_conv2d_9_kernel:=
/read_114_disablecopyonread_adam_m_conv2d_9_bias:=
/read_115_disablecopyonread_adam_v_conv2d_9_bias:*
 read_116_disablecopyonread_total: *
 read_117_disablecopyonread_count: 
savev2_const
identity_237ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_100/DisableCopyOnReadЂRead_100/ReadVariableOpЂRead_101/DisableCopyOnReadЂRead_101/ReadVariableOpЂRead_102/DisableCopyOnReadЂRead_102/ReadVariableOpЂRead_103/DisableCopyOnReadЂRead_103/ReadVariableOpЂRead_104/DisableCopyOnReadЂRead_104/ReadVariableOpЂRead_105/DisableCopyOnReadЂRead_105/ReadVariableOpЂRead_106/DisableCopyOnReadЂRead_106/ReadVariableOpЂRead_107/DisableCopyOnReadЂRead_107/ReadVariableOpЂRead_108/DisableCopyOnReadЂRead_108/ReadVariableOpЂRead_109/DisableCopyOnReadЂRead_109/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_110/DisableCopyOnReadЂRead_110/ReadVariableOpЂRead_111/DisableCopyOnReadЂRead_111/ReadVariableOpЂRead_112/DisableCopyOnReadЂRead_112/ReadVariableOpЂRead_113/DisableCopyOnReadЂRead_113/ReadVariableOpЂRead_114/DisableCopyOnReadЂRead_114/ReadVariableOpЂRead_115/DisableCopyOnReadЂRead_115/ReadVariableOpЂRead_116/DisableCopyOnReadЂRead_116/ReadVariableOpЂRead_117/DisableCopyOnReadЂRead_117/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_52/DisableCopyOnReadЂRead_52/ReadVariableOpЂRead_53/DisableCopyOnReadЂRead_53/ReadVariableOpЂRead_54/DisableCopyOnReadЂRead_54/ReadVariableOpЂRead_55/DisableCopyOnReadЂRead_55/ReadVariableOpЂRead_56/DisableCopyOnReadЂRead_56/ReadVariableOpЂRead_57/DisableCopyOnReadЂRead_57/ReadVariableOpЂRead_58/DisableCopyOnReadЂRead_58/ReadVariableOpЂRead_59/DisableCopyOnReadЂRead_59/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_60/DisableCopyOnReadЂRead_60/ReadVariableOpЂRead_61/DisableCopyOnReadЂRead_61/ReadVariableOpЂRead_62/DisableCopyOnReadЂRead_62/ReadVariableOpЂRead_63/DisableCopyOnReadЂRead_63/ReadVariableOpЂRead_64/DisableCopyOnReadЂRead_64/ReadVariableOpЂRead_65/DisableCopyOnReadЂRead_65/ReadVariableOpЂRead_66/DisableCopyOnReadЂRead_66/ReadVariableOpЂRead_67/DisableCopyOnReadЂRead_67/ReadVariableOpЂRead_68/DisableCopyOnReadЂRead_68/ReadVariableOpЂRead_69/DisableCopyOnReadЂRead_69/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_70/DisableCopyOnReadЂRead_70/ReadVariableOpЂRead_71/DisableCopyOnReadЂRead_71/ReadVariableOpЂRead_72/DisableCopyOnReadЂRead_72/ReadVariableOpЂRead_73/DisableCopyOnReadЂRead_73/ReadVariableOpЂRead_74/DisableCopyOnReadЂRead_74/ReadVariableOpЂRead_75/DisableCopyOnReadЂRead_75/ReadVariableOpЂRead_76/DisableCopyOnReadЂRead_76/ReadVariableOpЂRead_77/DisableCopyOnReadЂRead_77/ReadVariableOpЂRead_78/DisableCopyOnReadЂRead_78/ReadVariableOpЂRead_79/DisableCopyOnReadЂRead_79/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_80/DisableCopyOnReadЂRead_80/ReadVariableOpЂRead_81/DisableCopyOnReadЂRead_81/ReadVariableOpЂRead_82/DisableCopyOnReadЂRead_82/ReadVariableOpЂRead_83/DisableCopyOnReadЂRead_83/ReadVariableOpЂRead_84/DisableCopyOnReadЂRead_84/ReadVariableOpЂRead_85/DisableCopyOnReadЂRead_85/ReadVariableOpЂRead_86/DisableCopyOnReadЂRead_86/ReadVariableOpЂRead_87/DisableCopyOnReadЂRead_87/ReadVariableOpЂRead_88/DisableCopyOnReadЂRead_88/ReadVariableOpЂRead_89/DisableCopyOnReadЂRead_89/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpЂRead_90/DisableCopyOnReadЂRead_90/ReadVariableOpЂRead_91/DisableCopyOnReadЂRead_91/ReadVariableOpЂRead_92/DisableCopyOnReadЂRead_92/ReadVariableOpЂRead_93/DisableCopyOnReadЂRead_93/ReadVariableOpЂRead_94/DisableCopyOnReadЂRead_94/ReadVariableOpЂRead_95/DisableCopyOnReadЂRead_95/ReadVariableOpЂRead_96/DisableCopyOnReadЂRead_96/ReadVariableOpЂRead_97/DisableCopyOnReadЂRead_97/ReadVariableOpЂRead_98/DisableCopyOnReadЂRead_98/ReadVariableOpЂRead_99/DisableCopyOnReadЂRead_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 Ј
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
  
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_2/DisableCopyOnReadDisableCopyOnRead2read_2_disablecopyonread_group_normalization_gamma"/device:CPU:0*
_output_shapes
 Ў
Read_2/ReadVariableOpReadVariableOp2read_2_disablecopyonread_group_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_group_normalization_beta"/device:CPU:0*
_output_shapes
 ­
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_group_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 А
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_6/DisableCopyOnReadDisableCopyOnRead4read_6_disablecopyonread_group_normalization_1_gamma"/device:CPU:0*
_output_shapes
 А
Read_6/ReadVariableOpReadVariableOp4read_6_disablecopyonread_group_normalization_1_gamma^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_7/DisableCopyOnReadDisableCopyOnRead3read_7_disablecopyonread_group_normalization_1_beta"/device:CPU:0*
_output_shapes
 Џ
Read_7/ReadVariableOpReadVariableOp3read_7_disablecopyonread_group_normalization_1_beta^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 А
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_conv2d_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_conv2d_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_group_normalization_2_gamma"/device:CPU:0*
_output_shapes
 Г
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_group_normalization_2_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_11/DisableCopyOnReadDisableCopyOnRead4read_11_disablecopyonread_group_normalization_2_beta"/device:CPU:0*
_output_shapes
 В
Read_11/ReadVariableOpReadVariableOp4read_11_disablecopyonread_group_normalization_2_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 Г
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv2d_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv2d_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_group_normalization_3_gamma"/device:CPU:0*
_output_shapes
 Г
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_group_normalization_3_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_group_normalization_3_beta"/device:CPU:0*
_output_shapes
 В
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_group_normalization_3_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 Г
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_conv2d_4_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
: @|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_conv2d_4_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_18/DisableCopyOnReadDisableCopyOnRead5read_18_disablecopyonread_group_normalization_4_gamma"/device:CPU:0*
_output_shapes
 Г
Read_18/ReadVariableOpReadVariableOp5read_18_disablecopyonread_group_normalization_4_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_group_normalization_4_beta"/device:CPU:0*
_output_shapes
 В
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_group_normalization_4_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 Г
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_conv2d_5_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ |
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_conv2d_5_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_22/DisableCopyOnReadDisableCopyOnRead5read_22_disablecopyonread_group_normalization_5_gamma"/device:CPU:0*
_output_shapes
 Г
Read_22/ReadVariableOpReadVariableOp5read_22_disablecopyonread_group_normalization_5_gamma^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_23/DisableCopyOnReadDisableCopyOnRead4read_23_disablecopyonread_group_normalization_5_beta"/device:CPU:0*
_output_shapes
 В
Read_23/ReadVariableOpReadVariableOp4read_23_disablecopyonread_group_normalization_5_beta^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 Г
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_conv2d_6_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:@|
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_conv2d_6_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_26/DisableCopyOnReadDisableCopyOnRead5read_26_disablecopyonread_group_normalization_6_gamma"/device:CPU:0*
_output_shapes
 Г
Read_26/ReadVariableOpReadVariableOp5read_26_disablecopyonread_group_normalization_6_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_27/DisableCopyOnReadDisableCopyOnRead4read_27_disablecopyonread_group_normalization_6_beta"/device:CPU:0*
_output_shapes
 В
Read_27/ReadVariableOpReadVariableOp4read_27_disablecopyonread_group_normalization_6_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 Г
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_conv2d_7_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
: |
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_conv2d_7_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_30/DisableCopyOnReadDisableCopyOnRead5read_30_disablecopyonread_group_normalization_7_gamma"/device:CPU:0*
_output_shapes
 Г
Read_30/ReadVariableOpReadVariableOp5read_30_disablecopyonread_group_normalization_7_gamma^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_31/DisableCopyOnReadDisableCopyOnRead4read_31_disablecopyonread_group_normalization_7_beta"/device:CPU:0*
_output_shapes
 В
Read_31/ReadVariableOpReadVariableOp4read_31_disablecopyonread_group_normalization_7_beta^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_32/DisableCopyOnReadDisableCopyOnRead)read_32_disablecopyonread_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 Г
Read_32/ReadVariableOpReadVariableOp)read_32_disablecopyonread_conv2d_8_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_33/DisableCopyOnReadDisableCopyOnRead'read_33_disablecopyonread_conv2d_8_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_33/ReadVariableOpReadVariableOp'read_33_disablecopyonread_conv2d_8_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_34/DisableCopyOnReadDisableCopyOnRead5read_34_disablecopyonread_group_normalization_8_gamma"/device:CPU:0*
_output_shapes
 Г
Read_34/ReadVariableOpReadVariableOp5read_34_disablecopyonread_group_normalization_8_gamma^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_35/DisableCopyOnReadDisableCopyOnRead4read_35_disablecopyonread_group_normalization_8_beta"/device:CPU:0*
_output_shapes
 В
Read_35/ReadVariableOpReadVariableOp4read_35_disablecopyonread_group_normalization_8_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_36/DisableCopyOnReadDisableCopyOnRead)read_36_disablecopyonread_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 Г
Read_36/ReadVariableOpReadVariableOp)read_36_disablecopyonread_conv2d_9_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_37/DisableCopyOnReadDisableCopyOnRead'read_37_disablecopyonread_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_37/ReadVariableOpReadVariableOp'read_37_disablecopyonread_conv2d_9_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_38/DisableCopyOnReadDisableCopyOnRead#read_38_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_38/ReadVariableOpReadVariableOp#read_38_disablecopyonread_iteration^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_39/DisableCopyOnReadDisableCopyOnRead'read_39_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_39/ReadVariableOpReadVariableOp'read_39_disablecopyonread_learning_rate^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_conv2d_kernel"/device:CPU:0*
_output_shapes
 И
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_conv2d_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_conv2d_kernel"/device:CPU:0*
_output_shapes
 И
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_conv2d_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_42/DisableCopyOnReadDisableCopyOnRead,read_42_disablecopyonread_adam_m_conv2d_bias"/device:CPU:0*
_output_shapes
 Њ
Read_42/ReadVariableOpReadVariableOp,read_42_disablecopyonread_adam_m_conv2d_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_43/DisableCopyOnReadDisableCopyOnRead,read_43_disablecopyonread_adam_v_conv2d_bias"/device:CPU:0*
_output_shapes
 Њ
Read_43/ReadVariableOpReadVariableOp,read_43_disablecopyonread_adam_v_conv2d_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_44/DisableCopyOnReadDisableCopyOnRead:read_44_disablecopyonread_adam_m_group_normalization_gamma"/device:CPU:0*
_output_shapes
 И
Read_44/ReadVariableOpReadVariableOp:read_44_disablecopyonread_adam_m_group_normalization_gamma^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_45/DisableCopyOnReadDisableCopyOnRead:read_45_disablecopyonread_adam_v_group_normalization_gamma"/device:CPU:0*
_output_shapes
 И
Read_45/ReadVariableOpReadVariableOp:read_45_disablecopyonread_adam_v_group_normalization_gamma^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_46/DisableCopyOnReadDisableCopyOnRead9read_46_disablecopyonread_adam_m_group_normalization_beta"/device:CPU:0*
_output_shapes
 З
Read_46/ReadVariableOpReadVariableOp9read_46_disablecopyonread_adam_m_group_normalization_beta^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_47/DisableCopyOnReadDisableCopyOnRead9read_47_disablecopyonread_adam_v_group_normalization_beta"/device:CPU:0*
_output_shapes
 З
Read_47/ReadVariableOpReadVariableOp9read_47_disablecopyonread_adam_v_group_normalization_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 К
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_conv2d_1_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 К
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_conv2d_1_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_conv2d_1_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_conv2d_1_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_52/DisableCopyOnReadDisableCopyOnRead<read_52_disablecopyonread_adam_m_group_normalization_1_gamma"/device:CPU:0*
_output_shapes
 К
Read_52/ReadVariableOpReadVariableOp<read_52_disablecopyonread_adam_m_group_normalization_1_gamma^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_53/DisableCopyOnReadDisableCopyOnRead<read_53_disablecopyonread_adam_v_group_normalization_1_gamma"/device:CPU:0*
_output_shapes
 К
Read_53/ReadVariableOpReadVariableOp<read_53_disablecopyonread_adam_v_group_normalization_1_gamma^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_54/DisableCopyOnReadDisableCopyOnRead;read_54_disablecopyonread_adam_m_group_normalization_1_beta"/device:CPU:0*
_output_shapes
 Й
Read_54/ReadVariableOpReadVariableOp;read_54_disablecopyonread_adam_m_group_normalization_1_beta^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_55/DisableCopyOnReadDisableCopyOnRead;read_55_disablecopyonread_adam_v_group_normalization_1_beta"/device:CPU:0*
_output_shapes
 Й
Read_55/ReadVariableOpReadVariableOp;read_55_disablecopyonread_adam_v_group_normalization_1_beta^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_56/DisableCopyOnReadDisableCopyOnRead0read_56_disablecopyonread_adam_m_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 К
Read_56/ReadVariableOpReadVariableOp0read_56_disablecopyonread_adam_m_conv2d_2_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_57/DisableCopyOnReadDisableCopyOnRead0read_57_disablecopyonread_adam_v_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 К
Read_57/ReadVariableOpReadVariableOp0read_57_disablecopyonread_adam_v_conv2d_2_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_58/DisableCopyOnReadDisableCopyOnRead.read_58_disablecopyonread_adam_m_conv2d_2_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_58/ReadVariableOpReadVariableOp.read_58_disablecopyonread_adam_m_conv2d_2_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_59/DisableCopyOnReadDisableCopyOnRead.read_59_disablecopyonread_adam_v_conv2d_2_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_59/ReadVariableOpReadVariableOp.read_59_disablecopyonread_adam_v_conv2d_2_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_60/DisableCopyOnReadDisableCopyOnRead<read_60_disablecopyonread_adam_m_group_normalization_2_gamma"/device:CPU:0*
_output_shapes
 К
Read_60/ReadVariableOpReadVariableOp<read_60_disablecopyonread_adam_m_group_normalization_2_gamma^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_61/DisableCopyOnReadDisableCopyOnRead<read_61_disablecopyonread_adam_v_group_normalization_2_gamma"/device:CPU:0*
_output_shapes
 К
Read_61/ReadVariableOpReadVariableOp<read_61_disablecopyonread_adam_v_group_normalization_2_gamma^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_62/DisableCopyOnReadDisableCopyOnRead;read_62_disablecopyonread_adam_m_group_normalization_2_beta"/device:CPU:0*
_output_shapes
 Й
Read_62/ReadVariableOpReadVariableOp;read_62_disablecopyonread_adam_m_group_normalization_2_beta^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_63/DisableCopyOnReadDisableCopyOnRead;read_63_disablecopyonread_adam_v_group_normalization_2_beta"/device:CPU:0*
_output_shapes
 Й
Read_63/ReadVariableOpReadVariableOp;read_63_disablecopyonread_adam_v_group_normalization_2_beta^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_64/DisableCopyOnReadDisableCopyOnRead0read_64_disablecopyonread_adam_m_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 К
Read_64/ReadVariableOpReadVariableOp0read_64_disablecopyonread_adam_m_conv2d_3_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_65/DisableCopyOnReadDisableCopyOnRead0read_65_disablecopyonread_adam_v_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 К
Read_65/ReadVariableOpReadVariableOp0read_65_disablecopyonread_adam_v_conv2d_3_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_66/DisableCopyOnReadDisableCopyOnRead.read_66_disablecopyonread_adam_m_conv2d_3_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_66/ReadVariableOpReadVariableOp.read_66_disablecopyonread_adam_m_conv2d_3_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_67/DisableCopyOnReadDisableCopyOnRead.read_67_disablecopyonread_adam_v_conv2d_3_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_67/ReadVariableOpReadVariableOp.read_67_disablecopyonread_adam_v_conv2d_3_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_68/DisableCopyOnReadDisableCopyOnRead<read_68_disablecopyonread_adam_m_group_normalization_3_gamma"/device:CPU:0*
_output_shapes
 К
Read_68/ReadVariableOpReadVariableOp<read_68_disablecopyonread_adam_m_group_normalization_3_gamma^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_69/DisableCopyOnReadDisableCopyOnRead<read_69_disablecopyonread_adam_v_group_normalization_3_gamma"/device:CPU:0*
_output_shapes
 К
Read_69/ReadVariableOpReadVariableOp<read_69_disablecopyonread_adam_v_group_normalization_3_gamma^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_70/DisableCopyOnReadDisableCopyOnRead;read_70_disablecopyonread_adam_m_group_normalization_3_beta"/device:CPU:0*
_output_shapes
 Й
Read_70/ReadVariableOpReadVariableOp;read_70_disablecopyonread_adam_m_group_normalization_3_beta^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_71/DisableCopyOnReadDisableCopyOnRead;read_71_disablecopyonread_adam_v_group_normalization_3_beta"/device:CPU:0*
_output_shapes
 Й
Read_71/ReadVariableOpReadVariableOp;read_71_disablecopyonread_adam_v_group_normalization_3_beta^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_72/DisableCopyOnReadDisableCopyOnRead0read_72_disablecopyonread_adam_m_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 К
Read_72/ReadVariableOpReadVariableOp0read_72_disablecopyonread_adam_m_conv2d_4_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0x
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_73/DisableCopyOnReadDisableCopyOnRead0read_73_disablecopyonread_adam_v_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 К
Read_73/ReadVariableOpReadVariableOp0read_73_disablecopyonread_adam_v_conv2d_4_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0x
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_74/DisableCopyOnReadDisableCopyOnRead.read_74_disablecopyonread_adam_m_conv2d_4_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_74/ReadVariableOpReadVariableOp.read_74_disablecopyonread_adam_m_conv2d_4_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_75/DisableCopyOnReadDisableCopyOnRead.read_75_disablecopyonread_adam_v_conv2d_4_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_75/ReadVariableOpReadVariableOp.read_75_disablecopyonread_adam_v_conv2d_4_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_76/DisableCopyOnReadDisableCopyOnRead<read_76_disablecopyonread_adam_m_group_normalization_4_gamma"/device:CPU:0*
_output_shapes
 К
Read_76/ReadVariableOpReadVariableOp<read_76_disablecopyonread_adam_m_group_normalization_4_gamma^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_77/DisableCopyOnReadDisableCopyOnRead<read_77_disablecopyonread_adam_v_group_normalization_4_gamma"/device:CPU:0*
_output_shapes
 К
Read_77/ReadVariableOpReadVariableOp<read_77_disablecopyonread_adam_v_group_normalization_4_gamma^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_78/DisableCopyOnReadDisableCopyOnRead;read_78_disablecopyonread_adam_m_group_normalization_4_beta"/device:CPU:0*
_output_shapes
 Й
Read_78/ReadVariableOpReadVariableOp;read_78_disablecopyonread_adam_m_group_normalization_4_beta^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_79/DisableCopyOnReadDisableCopyOnRead;read_79_disablecopyonread_adam_v_group_normalization_4_beta"/device:CPU:0*
_output_shapes
 Й
Read_79/ReadVariableOpReadVariableOp;read_79_disablecopyonread_adam_v_group_normalization_4_beta^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_80/DisableCopyOnReadDisableCopyOnRead0read_80_disablecopyonread_adam_m_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 К
Read_80/ReadVariableOpReadVariableOp0read_80_disablecopyonread_adam_m_conv2d_5_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0x
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ o
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ 
Read_81/DisableCopyOnReadDisableCopyOnRead0read_81_disablecopyonread_adam_v_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 К
Read_81/ReadVariableOpReadVariableOp0read_81_disablecopyonread_adam_v_conv2d_5_kernel^Read_81/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0x
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ o
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ 
Read_82/DisableCopyOnReadDisableCopyOnRead.read_82_disablecopyonread_adam_m_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_82/ReadVariableOpReadVariableOp.read_82_disablecopyonread_adam_m_conv2d_5_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_83/DisableCopyOnReadDisableCopyOnRead.read_83_disablecopyonread_adam_v_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_83/ReadVariableOpReadVariableOp.read_83_disablecopyonread_adam_v_conv2d_5_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_84/DisableCopyOnReadDisableCopyOnRead<read_84_disablecopyonread_adam_m_group_normalization_5_gamma"/device:CPU:0*
_output_shapes
 К
Read_84/ReadVariableOpReadVariableOp<read_84_disablecopyonread_adam_m_group_normalization_5_gamma^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_85/DisableCopyOnReadDisableCopyOnRead<read_85_disablecopyonread_adam_v_group_normalization_5_gamma"/device:CPU:0*
_output_shapes
 К
Read_85/ReadVariableOpReadVariableOp<read_85_disablecopyonread_adam_v_group_normalization_5_gamma^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_86/DisableCopyOnReadDisableCopyOnRead;read_86_disablecopyonread_adam_m_group_normalization_5_beta"/device:CPU:0*
_output_shapes
 Й
Read_86/ReadVariableOpReadVariableOp;read_86_disablecopyonread_adam_m_group_normalization_5_beta^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_87/DisableCopyOnReadDisableCopyOnRead;read_87_disablecopyonread_adam_v_group_normalization_5_beta"/device:CPU:0*
_output_shapes
 Й
Read_87/ReadVariableOpReadVariableOp;read_87_disablecopyonread_adam_v_group_normalization_5_beta^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_88/DisableCopyOnReadDisableCopyOnRead0read_88_disablecopyonread_adam_m_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 К
Read_88/ReadVariableOpReadVariableOp0read_88_disablecopyonread_adam_m_conv2d_6_kernel^Read_88/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*&
_output_shapes
:@
Read_89/DisableCopyOnReadDisableCopyOnRead0read_89_disablecopyonread_adam_v_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 К
Read_89/ReadVariableOpReadVariableOp0read_89_disablecopyonread_adam_v_conv2d_6_kernel^Read_89/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*&
_output_shapes
:@
Read_90/DisableCopyOnReadDisableCopyOnRead.read_90_disablecopyonread_adam_m_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_90/ReadVariableOpReadVariableOp.read_90_disablecopyonread_adam_m_conv2d_6_bias^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_91/DisableCopyOnReadDisableCopyOnRead.read_91_disablecopyonread_adam_v_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_91/ReadVariableOpReadVariableOp.read_91_disablecopyonread_adam_v_conv2d_6_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_92/DisableCopyOnReadDisableCopyOnRead<read_92_disablecopyonread_adam_m_group_normalization_6_gamma"/device:CPU:0*
_output_shapes
 К
Read_92/ReadVariableOpReadVariableOp<read_92_disablecopyonread_adam_m_group_normalization_6_gamma^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_93/DisableCopyOnReadDisableCopyOnRead<read_93_disablecopyonread_adam_v_group_normalization_6_gamma"/device:CPU:0*
_output_shapes
 К
Read_93/ReadVariableOpReadVariableOp<read_93_disablecopyonread_adam_v_group_normalization_6_gamma^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_94/DisableCopyOnReadDisableCopyOnRead;read_94_disablecopyonread_adam_m_group_normalization_6_beta"/device:CPU:0*
_output_shapes
 Й
Read_94/ReadVariableOpReadVariableOp;read_94_disablecopyonread_adam_m_group_normalization_6_beta^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_95/DisableCopyOnReadDisableCopyOnRead;read_95_disablecopyonread_adam_v_group_normalization_6_beta"/device:CPU:0*
_output_shapes
 Й
Read_95/ReadVariableOpReadVariableOp;read_95_disablecopyonread_adam_v_group_normalization_6_beta^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_96/DisableCopyOnReadDisableCopyOnRead0read_96_disablecopyonread_adam_m_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 К
Read_96/ReadVariableOpReadVariableOp0read_96_disablecopyonread_adam_m_conv2d_7_kernel^Read_96/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_97/DisableCopyOnReadDisableCopyOnRead0read_97_disablecopyonread_adam_v_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 К
Read_97/ReadVariableOpReadVariableOp0read_97_disablecopyonread_adam_v_conv2d_7_kernel^Read_97/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_98/DisableCopyOnReadDisableCopyOnRead.read_98_disablecopyonread_adam_m_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_98/ReadVariableOpReadVariableOp.read_98_disablecopyonread_adam_m_conv2d_7_bias^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_99/DisableCopyOnReadDisableCopyOnRead.read_99_disablecopyonread_adam_v_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_99/ReadVariableOpReadVariableOp.read_99_disablecopyonread_adam_v_conv2d_7_bias^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_100/DisableCopyOnReadDisableCopyOnRead=read_100_disablecopyonread_adam_m_group_normalization_7_gamma"/device:CPU:0*
_output_shapes
 Н
Read_100/ReadVariableOpReadVariableOp=read_100_disablecopyonread_adam_m_group_normalization_7_gamma^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_101/DisableCopyOnReadDisableCopyOnRead=read_101_disablecopyonread_adam_v_group_normalization_7_gamma"/device:CPU:0*
_output_shapes
 Н
Read_101/ReadVariableOpReadVariableOp=read_101_disablecopyonread_adam_v_group_normalization_7_gamma^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_102/DisableCopyOnReadDisableCopyOnRead<read_102_disablecopyonread_adam_m_group_normalization_7_beta"/device:CPU:0*
_output_shapes
 М
Read_102/ReadVariableOpReadVariableOp<read_102_disablecopyonread_adam_m_group_normalization_7_beta^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_103/DisableCopyOnReadDisableCopyOnRead<read_103_disablecopyonread_adam_v_group_normalization_7_beta"/device:CPU:0*
_output_shapes
 М
Read_103/ReadVariableOpReadVariableOp<read_103_disablecopyonread_adam_v_group_normalization_7_beta^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_104/DisableCopyOnReadDisableCopyOnRead1read_104_disablecopyonread_adam_m_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 Н
Read_104/ReadVariableOpReadVariableOp1read_104_disablecopyonread_adam_m_conv2d_8_kernel^Read_104/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_105/DisableCopyOnReadDisableCopyOnRead1read_105_disablecopyonread_adam_v_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 Н
Read_105/ReadVariableOpReadVariableOp1read_105_disablecopyonread_adam_v_conv2d_8_kernel^Read_105/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_106/DisableCopyOnReadDisableCopyOnRead/read_106_disablecopyonread_adam_m_conv2d_8_bias"/device:CPU:0*
_output_shapes
 Џ
Read_106/ReadVariableOpReadVariableOp/read_106_disablecopyonread_adam_m_conv2d_8_bias^Read_106/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_107/DisableCopyOnReadDisableCopyOnRead/read_107_disablecopyonread_adam_v_conv2d_8_bias"/device:CPU:0*
_output_shapes
 Џ
Read_107/ReadVariableOpReadVariableOp/read_107_disablecopyonread_adam_v_conv2d_8_bias^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_108/DisableCopyOnReadDisableCopyOnRead=read_108_disablecopyonread_adam_m_group_normalization_8_gamma"/device:CPU:0*
_output_shapes
 Н
Read_108/ReadVariableOpReadVariableOp=read_108_disablecopyonread_adam_m_group_normalization_8_gamma^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_109/DisableCopyOnReadDisableCopyOnRead=read_109_disablecopyonread_adam_v_group_normalization_8_gamma"/device:CPU:0*
_output_shapes
 Н
Read_109/ReadVariableOpReadVariableOp=read_109_disablecopyonread_adam_v_group_normalization_8_gamma^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_110/DisableCopyOnReadDisableCopyOnRead<read_110_disablecopyonread_adam_m_group_normalization_8_beta"/device:CPU:0*
_output_shapes
 М
Read_110/ReadVariableOpReadVariableOp<read_110_disablecopyonread_adam_m_group_normalization_8_beta^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_111/DisableCopyOnReadDisableCopyOnRead<read_111_disablecopyonread_adam_v_group_normalization_8_beta"/device:CPU:0*
_output_shapes
 М
Read_111/ReadVariableOpReadVariableOp<read_111_disablecopyonread_adam_v_group_normalization_8_beta^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_112/DisableCopyOnReadDisableCopyOnRead1read_112_disablecopyonread_adam_m_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 Н
Read_112/ReadVariableOpReadVariableOp1read_112_disablecopyonread_adam_m_conv2d_9_kernel^Read_112/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_113/DisableCopyOnReadDisableCopyOnRead1read_113_disablecopyonread_adam_v_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 Н
Read_113/ReadVariableOpReadVariableOp1read_113_disablecopyonread_adam_v_conv2d_9_kernel^Read_113/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_114/DisableCopyOnReadDisableCopyOnRead/read_114_disablecopyonread_adam_m_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Џ
Read_114/ReadVariableOpReadVariableOp/read_114_disablecopyonread_adam_m_conv2d_9_bias^Read_114/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_115/DisableCopyOnReadDisableCopyOnRead/read_115_disablecopyonread_adam_v_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Џ
Read_115/ReadVariableOpReadVariableOp/read_115_disablecopyonread_adam_v_conv2d_9_bias^Read_115/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_116/DisableCopyOnReadDisableCopyOnRead read_116_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_116/ReadVariableOpReadVariableOp read_116_disablecopyonread_total^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_117/DisableCopyOnReadDisableCopyOnRead read_117_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_117/ReadVariableOpReadVariableOp read_117_disablecopyonread_count^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
: ї1
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:w*
dtype0* 1
value1B1wB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHо
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*
valueљBіwB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B В
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes{
y2w	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_236Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_237IdentityIdentity_236:output:0^NoOp*
T0*
_output_shapes
: Б1
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_237Identity_237:output:0*(
_construction_contextkEagerRuntime*
_input_shapesѓ
№: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv2d/kernel:+'
%
_user_specified_nameconv2d/bias:95
3
_user_specified_namegroup_normalization/gamma:84
2
_user_specified_namegroup_normalization/beta:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv2d_1/bias:;7
5
_user_specified_namegroup_normalization_1/gamma::6
4
_user_specified_namegroup_normalization_1/beta:/	+
)
_user_specified_nameconv2d_2/kernel:-
)
'
_user_specified_nameconv2d_2/bias:;7
5
_user_specified_namegroup_normalization_2/gamma::6
4
_user_specified_namegroup_normalization_2/beta:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_3/bias:;7
5
_user_specified_namegroup_normalization_3/gamma::6
4
_user_specified_namegroup_normalization_3/beta:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:;7
5
_user_specified_namegroup_normalization_4/gamma::6
4
_user_specified_namegroup_normalization_4/beta:/+
)
_user_specified_nameconv2d_5/kernel:-)
'
_user_specified_nameconv2d_5/bias:;7
5
_user_specified_namegroup_normalization_5/gamma::6
4
_user_specified_namegroup_normalization_5/beta:/+
)
_user_specified_nameconv2d_6/kernel:-)
'
_user_specified_nameconv2d_6/bias:;7
5
_user_specified_namegroup_normalization_6/gamma::6
4
_user_specified_namegroup_normalization_6/beta:/+
)
_user_specified_nameconv2d_7/kernel:-)
'
_user_specified_nameconv2d_7/bias:;7
5
_user_specified_namegroup_normalization_7/gamma:: 6
4
_user_specified_namegroup_normalization_7/beta:/!+
)
_user_specified_nameconv2d_8/kernel:-")
'
_user_specified_nameconv2d_8/bias:;#7
5
_user_specified_namegroup_normalization_8/gamma::$6
4
_user_specified_namegroup_normalization_8/beta:/%+
)
_user_specified_nameconv2d_9/kernel:-&)
'
_user_specified_nameconv2d_9/bias:)'%
#
_user_specified_name	iteration:-()
'
_user_specified_namelearning_rate:4)0
.
_user_specified_nameAdam/m/conv2d/kernel:4*0
.
_user_specified_nameAdam/v/conv2d/kernel:2+.
,
_user_specified_nameAdam/m/conv2d/bias:2,.
,
_user_specified_nameAdam/v/conv2d/bias:@-<
:
_user_specified_name" Adam/m/group_normalization/gamma:@.<
:
_user_specified_name" Adam/v/group_normalization/gamma:?/;
9
_user_specified_name!Adam/m/group_normalization/beta:?0;
9
_user_specified_name!Adam/v/group_normalization/beta:612
0
_user_specified_nameAdam/m/conv2d_1/kernel:622
0
_user_specified_nameAdam/v/conv2d_1/kernel:430
.
_user_specified_nameAdam/m/conv2d_1/bias:440
.
_user_specified_nameAdam/v/conv2d_1/bias:B5>
<
_user_specified_name$"Adam/m/group_normalization_1/gamma:B6>
<
_user_specified_name$"Adam/v/group_normalization_1/gamma:A7=
;
_user_specified_name#!Adam/m/group_normalization_1/beta:A8=
;
_user_specified_name#!Adam/v/group_normalization_1/beta:692
0
_user_specified_nameAdam/m/conv2d_2/kernel:6:2
0
_user_specified_nameAdam/v/conv2d_2/kernel:4;0
.
_user_specified_nameAdam/m/conv2d_2/bias:4<0
.
_user_specified_nameAdam/v/conv2d_2/bias:B=>
<
_user_specified_name$"Adam/m/group_normalization_2/gamma:B>>
<
_user_specified_name$"Adam/v/group_normalization_2/gamma:A?=
;
_user_specified_name#!Adam/m/group_normalization_2/beta:A@=
;
_user_specified_name#!Adam/v/group_normalization_2/beta:6A2
0
_user_specified_nameAdam/m/conv2d_3/kernel:6B2
0
_user_specified_nameAdam/v/conv2d_3/kernel:4C0
.
_user_specified_nameAdam/m/conv2d_3/bias:4D0
.
_user_specified_nameAdam/v/conv2d_3/bias:BE>
<
_user_specified_name$"Adam/m/group_normalization_3/gamma:BF>
<
_user_specified_name$"Adam/v/group_normalization_3/gamma:AG=
;
_user_specified_name#!Adam/m/group_normalization_3/beta:AH=
;
_user_specified_name#!Adam/v/group_normalization_3/beta:6I2
0
_user_specified_nameAdam/m/conv2d_4/kernel:6J2
0
_user_specified_nameAdam/v/conv2d_4/kernel:4K0
.
_user_specified_nameAdam/m/conv2d_4/bias:4L0
.
_user_specified_nameAdam/v/conv2d_4/bias:BM>
<
_user_specified_name$"Adam/m/group_normalization_4/gamma:BN>
<
_user_specified_name$"Adam/v/group_normalization_4/gamma:AO=
;
_user_specified_name#!Adam/m/group_normalization_4/beta:AP=
;
_user_specified_name#!Adam/v/group_normalization_4/beta:6Q2
0
_user_specified_nameAdam/m/conv2d_5/kernel:6R2
0
_user_specified_nameAdam/v/conv2d_5/kernel:4S0
.
_user_specified_nameAdam/m/conv2d_5/bias:4T0
.
_user_specified_nameAdam/v/conv2d_5/bias:BU>
<
_user_specified_name$"Adam/m/group_normalization_5/gamma:BV>
<
_user_specified_name$"Adam/v/group_normalization_5/gamma:AW=
;
_user_specified_name#!Adam/m/group_normalization_5/beta:AX=
;
_user_specified_name#!Adam/v/group_normalization_5/beta:6Y2
0
_user_specified_nameAdam/m/conv2d_6/kernel:6Z2
0
_user_specified_nameAdam/v/conv2d_6/kernel:4[0
.
_user_specified_nameAdam/m/conv2d_6/bias:4\0
.
_user_specified_nameAdam/v/conv2d_6/bias:B]>
<
_user_specified_name$"Adam/m/group_normalization_6/gamma:B^>
<
_user_specified_name$"Adam/v/group_normalization_6/gamma:A_=
;
_user_specified_name#!Adam/m/group_normalization_6/beta:A`=
;
_user_specified_name#!Adam/v/group_normalization_6/beta:6a2
0
_user_specified_nameAdam/m/conv2d_7/kernel:6b2
0
_user_specified_nameAdam/v/conv2d_7/kernel:4c0
.
_user_specified_nameAdam/m/conv2d_7/bias:4d0
.
_user_specified_nameAdam/v/conv2d_7/bias:Be>
<
_user_specified_name$"Adam/m/group_normalization_7/gamma:Bf>
<
_user_specified_name$"Adam/v/group_normalization_7/gamma:Ag=
;
_user_specified_name#!Adam/m/group_normalization_7/beta:Ah=
;
_user_specified_name#!Adam/v/group_normalization_7/beta:6i2
0
_user_specified_nameAdam/m/conv2d_8/kernel:6j2
0
_user_specified_nameAdam/v/conv2d_8/kernel:4k0
.
_user_specified_nameAdam/m/conv2d_8/bias:4l0
.
_user_specified_nameAdam/v/conv2d_8/bias:Bm>
<
_user_specified_name$"Adam/m/group_normalization_8/gamma:Bn>
<
_user_specified_name$"Adam/v/group_normalization_8/gamma:Ao=
;
_user_specified_name#!Adam/m/group_normalization_8/beta:Ap=
;
_user_specified_name#!Adam/v/group_normalization_8/beta:6q2
0
_user_specified_nameAdam/m/conv2d_9/kernel:6r2
0
_user_specified_nameAdam/v/conv2d_9/kernel:4s0
.
_user_specified_nameAdam/m/conv2d_9/bias:4t0
.
_user_specified_nameAdam/v/conv2d_9/bias:%u!

_user_specified_nametotal:%v!

_user_specified_namecount:=w9

_output_shapes
: 

_user_specified_nameConst
Ў
^
B__inference_re_lu_layer_call_and_return_conditional_losses_6466588

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
l
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6467527

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
`
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6466696

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_44431
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
з
M
1__inference_up_sampling2d_1_layer_call_fn_6468103

inputs
identityї
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6466273
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж
r
H__inference_concatenate_layer_call_and_return_conditional_losses_6466731

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@q
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
 
7__inference_group_normalization_2_layer_call_fn_6467652

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6465984
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467646:'#
!
_user_specified_name	6467648
Ђ
E
)__inference_re_lu_4_layer_call_fn_6467957

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6466696z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Г
ў
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6466778

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
п
Q
5__inference_average_pooling2d_4_layer_call_fn_6467870

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *Y
fTRR
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6466085
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_44541
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Г
ў
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6467998

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ
J
"__inference__update_step_xla_44496
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
А
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6466793

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6467979

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6466440

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Г
ў
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6466626

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
E
)__inference_re_lu_7_layer_call_fn_6468352

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6466793z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	

*__inference_conv2d_3_layer_call_fn_6467739

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6466653
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467733:'#
!
_user_specified_name	6467735
А
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6468357

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЮB

R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6468347

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ
J
"__inference__update_step_xla_44521
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_44446
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	

*__inference_conv2d_7_layer_call_fn_6468260

inputs!
unknown: 
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6466778
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468254:'#
!
_user_specified_name	6468256
Г
ў
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6466680

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Я
V
"__inference__update_step_xla_44511
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
г
K
/__inference_up_sampling2d_layer_call_fn_6467967

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6466179
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	

*__inference_conv2d_1_layer_call_fn_6467507

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6466599
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6467501:'#
!
_user_specified_name	6467503
Ђ
E
)__inference_re_lu_3_layer_call_fn_6467841

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6466669z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	

*__inference_conv2d_6_layer_call_fn_6468124

inputs!
unknown:@
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6466743
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468118:'#
!
_user_specified_name	6468120
ЬB

P__inference_group_normalization_layer_call_and_return_conditional_losses_6465810

inputs/
!reshape_1_readvariableop_resource:/
!reshape_2_readvariableop_resource:
identityЂReshape_1/ReadVariableOpЂReshape_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :d
floordivFloorDivstrided_slice_4:output:0floordiv/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ї
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0floordiv:z:0*
N*
T0*
_output_shapes
:z
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџs
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(y
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:џџџџџџџџџЏ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Г
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
	keep_dims(h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_5StridedSliceShape:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :h

floordiv_1FloorDivstrided_slice_5:output:0floordiv_1/y:output:0*
T0*
_output_shapes
: v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_2/shapePackReshape_2/shape/0:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0Reshape_2/shape/3:output:0floordiv_1:z:0*
N*
T0*
_output_shapes
:
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџi
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ~
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ{
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityReshape_3:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџX
NoOpNoOp^Reshape_1/ReadVariableOp^Reshape_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp24
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ
J
"__inference__update_step_xla_44426
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	
 
7__inference_group_normalization_6_layer_call_fn_6468143

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8 *[
fVRT
R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6466346
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	6468137:'#
!
_user_specified_name	6468139
Г
ў
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6466708

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ
J
"__inference__update_step_xla_44386
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Г
ў
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6468542

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ё
l
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6465911

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
ў
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6466599

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
А
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6468493

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ЪL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*п
serving_defaultЫ
U
input_1J
serving_default_input_1:0+џџџџџџџџџџџџџџџџџџџџџџџџџџџV
conv2d_9J
StatefulPartitionedCall:0+џџџџџџџџџџџџџџџџџџџџџџџџџџџtensorflow/serving/predict:ъд
Њ

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer-25
layer-26
layer_with_weights-12
layer-27
layer_with_weights-13
layer-28
layer-29
layer-30
 layer-31
!layer_with_weights-14
!layer-32
"layer_with_weights-15
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-16
&layer-37
'layer_with_weights-17
'layer-38
(layer-39
)layer-40
*layer-41
+layer_with_weights-18
+layer-42
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_default_save_signature
3	optimizer
4
signatures"
_tf_keras_network
"
_tf_keras_input_layer
н
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
К
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
	Jgamma
Kbeta"
_tf_keras_layer
Ѕ
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
н
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
К
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
	ggamma
hbeta"
_tf_keras_layer
Ѕ
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
н
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias
 w_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
Р
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

gamma
	beta"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

Ёgamma
	Ђbeta"
_tf_keras_layer
Ћ
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
Џkernel
	Аbias
!Б_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses

Оgamma
	Пbeta"
_tf_keras_layer
Ћ
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
вkernel
	гbias
!д_jit_compiled_convolution_op"
_tf_keras_layer
Т
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses

лgamma
	мbeta"
_tf_keras_layer
Ћ
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses
ѕkernel
	іbias
!ї_jit_compiled_convolution_op"
_tf_keras_layer
Т
ј	variables
љtrainable_variables
њregularization_losses
ћ	keras_api
ќ__call__
+§&call_and_return_all_conditional_losses

ўgamma
	џbeta"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
Т
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

Ёgamma
	Ђbeta"
_tf_keras_layer
Ћ
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Џ	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
Лkernel
	Мbias
!Н_jit_compiled_convolution_op"
_tf_keras_layer
Т
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses

Фgamma
	Хbeta"
_tf_keras_layer
Ћ
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses
оkernel
	пbias
!р_jit_compiled_convolution_op"
_tf_keras_layer
т
;0
<1
J2
K3
X4
Y5
g6
h7
u8
v9
10
11
12
13
Ё14
Ђ15
Џ16
А17
О18
П19
в20
г21
л22
м23
ѕ24
і25
ў26
џ27
28
29
Ё30
Ђ31
Л32
М33
Ф34
Х35
о36
п37"
trackable_list_wrapper
т
;0
<1
J2
K3
X4
Y5
g6
h7
u8
v9
10
11
12
13
Ё14
Ђ15
Џ16
А17
О18
П19
в20
г21
л22
м23
ѕ24
і25
ў26
џ27
28
29
Ё30
Ђ31
Л32
М33
Ф34
Х35
о36
п37"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
2_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Х
цtrace_0
чtrace_12
'__inference_model_layer_call_fn_6467058
'__inference_model_layer_call_fn_6467139Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zцtrace_0zчtrace_1
ћ
шtrace_0
щtrace_12Р
B__inference_model_layer_call_and_return_conditional_losses_6466855
B__inference_model_layer_call_and_return_conditional_losses_6466977Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0zщtrace_1
ЭBЪ
"__inference__wrapped_model_6465732input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ
ъ
_variables
ы_iterations
ь_learning_rate
э_index_dict
ю
_momentums
я_velocities
№_update_step_xla"
experimentalOptimizer
-
ёserving_default"
signature_map
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
ф
їtrace_02Х
(__inference_conv2d_layer_call_fn_6467391
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0
џ
јtrace_02р
C__inference_conv2d_layer_call_and_return_conditional_losses_6467401
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0
':%2conv2d/kernel
:2conv2d/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
я
ўtrace_02а
3__inference_average_pooling2d_layer_call_fn_6467406
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zўtrace_0

џtrace_02ы
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6467411
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
5__inference_group_normalization_layer_call_fn_6467420
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
P__inference_group_normalization_layer_call_and_return_conditional_losses_6467488
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
':%2group_normalization/gamma
&:$2group_normalization/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
у
trace_02Ф
'__inference_re_lu_layer_call_fn_6467493
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ў
trace_02п
B__inference_re_lu_layer_call_and_return_conditional_losses_6467498
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_conv2d_1_layer_call_fn_6467507
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02т
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467517
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):'2conv2d_1/kernel
:2conv2d_1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
5__inference_average_pooling2d_1_layer_call_fn_6467522
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6467527
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
ѓ
Ёtrace_02д
7__inference_group_normalization_1_layer_call_fn_6467536
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0

Ђtrace_02я
R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6467604
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЂtrace_0
):'2group_normalization_1/gamma
(:&2group_normalization_1/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ѓnon_trainable_variables
Єlayers
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
х
Јtrace_02Ц
)__inference_re_lu_1_layer_call_fn_6467609
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЈtrace_0

Љtrace_02с
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6467614
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
ц
Џtrace_02Ч
*__inference_conv2d_2_layer_call_fn_6467623
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЏtrace_0

Аtrace_02т
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6467633
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0
):'2conv2d_2/kernel
:2conv2d_2/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
ё
Жtrace_02в
5__inference_average_pooling2d_2_layer_call_fn_6467638
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0

Зtrace_02э
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6467643
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ж
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ѓ
Нtrace_02д
7__inference_group_normalization_2_layer_call_fn_6467652
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0

Оtrace_02я
R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6467720
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zОtrace_0
):'2group_normalization_2/gamma
(:&2group_normalization_2/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
Фtrace_02Ц
)__inference_re_lu_2_layer_call_fn_6467725
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0

Хtrace_02с
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6467730
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zХtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ц
Ыtrace_02Ч
*__inference_conv2d_3_layer_call_fn_6467739
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЫtrace_0

Ьtrace_02т
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6467749
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0
):' 2conv2d_3/kernel
: 2conv2d_3/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ё
вtrace_02в
5__inference_average_pooling2d_3_layer_call_fn_6467754
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zвtrace_0

гtrace_02э
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6467759
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0
0
Ё0
Ђ1"
trackable_list_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ѓ
йtrace_02д
7__inference_group_normalization_3_layer_call_fn_6467768
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0

кtrace_02я
R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6467836
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0
):' 2group_normalization_3/gamma
(:& 2group_normalization_3/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
х
рtrace_02Ц
)__inference_re_lu_3_layer_call_fn_6467841
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zрtrace_0

сtrace_02с
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6467846
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zсtrace_0
0
Џ0
А1"
trackable_list_wrapper
0
Џ0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
ц
чtrace_02Ч
*__inference_conv2d_4_layer_call_fn_6467855
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zчtrace_0

шtrace_02т
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6467865
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0
):' @2conv2d_4/kernel
:@2conv2d_4/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
ё
юtrace_02в
5__inference_average_pooling2d_4_layer_call_fn_6467870
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zюtrace_0

яtrace_02э
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6467875
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zяtrace_0
0
О0
П1"
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
ѓ
ѕtrace_02д
7__inference_group_normalization_4_layer_call_fn_6467884
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѕtrace_0

іtrace_02я
R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6467952
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zіtrace_0
):'@2group_normalization_4/gamma
(:&@2group_normalization_4/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
їnon_trainable_variables
јlayers
љmetrics
 њlayer_regularization_losses
ћlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
х
ќtrace_02Ц
)__inference_re_lu_4_layer_call_fn_6467957
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zќtrace_0

§trace_02с
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6467962
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z§trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ўnon_trainable_variables
џlayers
metrics
 layer_regularization_losses
layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
/__inference_up_sampling2d_layer_call_fn_6467967
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ч
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6467979
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
в0
г1"
trackable_list_wrapper
0
в0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_conv2d_5_layer_call_fn_6467988
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02т
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6467998
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):'@ 2conv2d_5/kernel
: 2conv2d_5/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
л0
м1"
trackable_list_wrapper
0
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
ѓ
trace_02д
7__inference_group_normalization_5_layer_call_fn_6468007
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02я
R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6468075
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):' 2group_normalization_5/gamma
(:& 2group_normalization_5/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_re_lu_5_layer_call_fn_6468080
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02с
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6468085
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
у	variables
фtrainable_variables
хregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_concatenate_layer_call_fn_6468091
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

 trace_02х
H__inference_concatenate_layer_call_and_return_conditional_losses_6468098
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
э
Іtrace_02Ю
1__inference_up_sampling2d_1_layer_call_fn_6468103
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0

Їtrace_02щ
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6468115
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЇtrace_0
0
ѕ0
і1"
trackable_list_wrapper
0
ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
ц
­trace_02Ч
*__inference_conv2d_6_layer_call_fn_6468124
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0

Ўtrace_02т
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6468134
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0
):'@2conv2d_6/kernel
:2conv2d_6/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
ў0
џ1"
trackable_list_wrapper
0
ў0
џ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
ј	variables
љtrainable_variables
њregularization_losses
ќ__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
ѓ
Дtrace_02д
7__inference_group_normalization_6_layer_call_fn_6468143
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0

Еtrace_02я
R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6468211
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
):'2group_normalization_6/gamma
(:&2group_normalization_6/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
Лtrace_02Ц
)__inference_re_lu_6_layer_call_fn_6468216
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0

Мtrace_02с
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6468221
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ы
Тtrace_02Ь
/__inference_concatenate_1_layer_call_fn_6468227
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0

Уtrace_02ч
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6468234
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
Щtrace_02Ю
1__inference_up_sampling2d_2_layer_call_fn_6468239
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЩtrace_0

Ъtrace_02щ
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6468251
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ц
аtrace_02Ч
*__inference_conv2d_7_layer_call_fn_6468260
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zаtrace_0

бtrace_02т
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6468270
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0
):' 2conv2d_7/kernel
:2conv2d_7/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
Ё0
Ђ1"
trackable_list_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ѓ
зtrace_02д
7__inference_group_normalization_7_layer_call_fn_6468279
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0

иtrace_02я
R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6468347
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zиtrace_0
):'2group_normalization_7/gamma
(:&2group_normalization_7/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
х
оtrace_02Ц
)__inference_re_lu_7_layer_call_fn_6468352
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zоtrace_0

пtrace_02с
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6468357
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zпtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
ы
хtrace_02Ь
/__inference_concatenate_2_layer_call_fn_6468363
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zхtrace_0

цtrace_02ч
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6468370
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zцtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
Џ	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
э
ьtrace_02Ю
1__inference_up_sampling2d_3_layer_call_fn_6468375
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zьtrace_0

эtrace_02щ
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6468387
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zэtrace_0
0
Л0
М1"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
юnon_trainable_variables
яlayers
№metrics
 ёlayer_regularization_losses
ђlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
ц
ѓtrace_02Ч
*__inference_conv2d_8_layer_call_fn_6468396
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѓtrace_0

єtrace_02т
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6468406
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0
):'2conv2d_8/kernel
:2conv2d_8/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
Ф0
Х1"
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
ѓ
њtrace_02д
7__inference_group_normalization_8_layer_call_fn_6468415
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zњtrace_0

ћtrace_02я
R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6468483
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zћtrace_0
):'2group_normalization_8/gamma
(:&2group_normalization_8/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ќnon_trainable_variables
§layers
ўmetrics
 џlayer_regularization_losses
layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_re_lu_8_layer_call_fn_6468488
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02с
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6468493
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
/__inference_concatenate_3_layer_call_fn_6468499
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ч
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6468506
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
1__inference_up_sampling2d_4_layer_call_fn_6468511
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6468523
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
о0
п1"
trackable_list_wrapper
0
о0
п1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_conv2d_9_layer_call_fn_6468532
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02т
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6468542
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):'2conv2d_9/kernel
:2conv2d_9/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яBь
'__inference_model_layer_call_fn_6467058input_1"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
'__inference_model_layer_call_fn_6467139input_1"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_layer_call_and_return_conditional_losses_6466855input_1"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_layer_call_and_return_conditional_losses_6466977input_1"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ы
ы0
1
2
3
4
5
6
7
 8
Ё9
Ђ10
Ѓ11
Є12
Ѕ13
І14
Ї15
Ј16
Љ17
Њ18
Ћ19
Ќ20
­21
Ў22
Џ23
А24
Б25
В26
Г27
Д28
Е29
Ж30
З31
И32
Й33
К34
Л35
М36
Н37
О38
П39
Р40
С41
Т42
У43
Ф44
Х45
Ц46
Ч47
Ш48
Щ49
Ъ50
Ы51
Ь52
Э53
Ю54
Я55
а56
б57
в58
г59
д60
е61
ж62
з63
и64
й65
к66
л67
м68
н69
о70
п71
р72
с73
т74
у75
ф76"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
ь
0
1
2
3
Ё4
Ѓ5
Ѕ6
Ї7
Љ8
Ћ9
­10
Џ11
Б12
Г13
Е14
З15
Й16
Л17
Н18
П19
С20
У21
Х22
Ч23
Щ24
Ы25
Э26
Я27
б28
г29
е30
з31
й32
л33
н34
п35
с36
у37"
trackable_list_wrapper
ь
0
1
2
 3
Ђ4
Є5
І6
Ј7
Њ8
Ќ9
Ў10
А11
В12
Д13
Ж14
И15
К16
М17
О18
Р19
Т20
Ф21
Ц22
Ш23
Ъ24
Ь25
Ю26
а27
в28
д29
ж30
и31
к32
м33
о34
р35
т36
ф37"
trackable_list_wrapper
э
хtrace_0
цtrace_1
чtrace_2
шtrace_3
щtrace_4
ъtrace_5
ыtrace_6
ьtrace_7
эtrace_8
юtrace_9
яtrace_10
№trace_11
ёtrace_12
ђtrace_13
ѓtrace_14
єtrace_15
ѕtrace_16
іtrace_17
їtrace_18
јtrace_19
љtrace_20
њtrace_21
ћtrace_22
ќtrace_23
§trace_24
ўtrace_25
џtrace_26
trace_27
trace_28
trace_29
trace_30
trace_31
trace_32
trace_33
trace_34
trace_35
trace_36
trace_372
"__inference__update_step_xla_44371
"__inference__update_step_xla_44376
"__inference__update_step_xla_44381
"__inference__update_step_xla_44386
"__inference__update_step_xla_44391
"__inference__update_step_xla_44396
"__inference__update_step_xla_44401
"__inference__update_step_xla_44406
"__inference__update_step_xla_44411
"__inference__update_step_xla_44416
"__inference__update_step_xla_44421
"__inference__update_step_xla_44426
"__inference__update_step_xla_44431
"__inference__update_step_xla_44436
"__inference__update_step_xla_44441
"__inference__update_step_xla_44446
"__inference__update_step_xla_44451
"__inference__update_step_xla_44456
"__inference__update_step_xla_44461
"__inference__update_step_xla_44466
"__inference__update_step_xla_44471
"__inference__update_step_xla_44476
"__inference__update_step_xla_44481
"__inference__update_step_xla_44486
"__inference__update_step_xla_44491
"__inference__update_step_xla_44496
"__inference__update_step_xla_44501
"__inference__update_step_xla_44506
"__inference__update_step_xla_44511
"__inference__update_step_xla_44516
"__inference__update_step_xla_44521
"__inference__update_step_xla_44526
"__inference__update_step_xla_44531
"__inference__update_step_xla_44536
"__inference__update_step_xla_44541
"__inference__update_step_xla_44546
"__inference__update_step_xla_44551
"__inference__update_step_xla_44556Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zхtrace_0zцtrace_1zчtrace_2zшtrace_3zщtrace_4zъtrace_5zыtrace_6zьtrace_7zэtrace_8zюtrace_9zяtrace_10z№trace_11zёtrace_12zђtrace_13zѓtrace_14zєtrace_15zѕtrace_16zіtrace_17zїtrace_18zјtrace_19zљtrace_20zњtrace_21zћtrace_22zќtrace_23z§trace_24zўtrace_25zџtrace_26ztrace_27ztrace_28ztrace_29ztrace_30ztrace_31ztrace_32ztrace_33ztrace_34ztrace_35ztrace_36ztrace_37
ЬBЩ
%__inference_signature_wrapper_6467382input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv2d_layer_call_fn_6467391inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv2d_layer_call_and_return_conditional_losses_6467401inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_average_pooling2d_layer_call_fn_6467406inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6467411inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_group_normalization_layer_call_fn_6467420inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_group_normalization_layer_call_and_return_conditional_losses_6467488inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_re_lu_layer_call_fn_6467493inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_re_lu_layer_call_and_return_conditional_losses_6467498inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_1_layer_call_fn_6467507inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467517inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_average_pooling2d_1_layer_call_fn_6467522inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6467527inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_1_layer_call_fn_6467536inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6467604inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_1_layer_call_fn_6467609inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6467614inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_2_layer_call_fn_6467623inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6467633inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_average_pooling2d_2_layer_call_fn_6467638inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6467643inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_2_layer_call_fn_6467652inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6467720inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_2_layer_call_fn_6467725inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6467730inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_3_layer_call_fn_6467739inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6467749inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_average_pooling2d_3_layer_call_fn_6467754inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6467759inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_3_layer_call_fn_6467768inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6467836inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_3_layer_call_fn_6467841inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6467846inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_4_layer_call_fn_6467855inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6467865inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_average_pooling2d_4_layer_call_fn_6467870inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6467875inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_4_layer_call_fn_6467884inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6467952inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_4_layer_call_fn_6467957inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6467962inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_up_sampling2d_layer_call_fn_6467967inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6467979inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_5_layer_call_fn_6467988inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6467998inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_5_layer_call_fn_6468007inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6468075inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_5_layer_call_fn_6468080inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6468085inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
-__inference_concatenate_layer_call_fn_6468091inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
H__inference_concatenate_layer_call_and_return_conditional_losses_6468098inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_up_sampling2d_1_layer_call_fn_6468103inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6468115inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_6_layer_call_fn_6468124inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6468134inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_6_layer_call_fn_6468143inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6468211inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_6_layer_call_fn_6468216inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6468221inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
/__inference_concatenate_1_layer_call_fn_6468227inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6468234inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_up_sampling2d_2_layer_call_fn_6468239inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6468251inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_7_layer_call_fn_6468260inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6468270inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_7_layer_call_fn_6468279inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6468347inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_7_layer_call_fn_6468352inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6468357inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
/__inference_concatenate_2_layer_call_fn_6468363inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6468370inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_up_sampling2d_3_layer_call_fn_6468375inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6468387inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_8_layer_call_fn_6468396inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6468406inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_group_normalization_8_layer_call_fn_6468415inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6468483inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_re_lu_8_layer_call_fn_6468488inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6468493inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
/__inference_concatenate_3_layer_call_fn_6468499inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6468506inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_up_sampling2d_4_layer_call_fn_6468511inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6468523inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv2d_9_layer_call_fn_6468532inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6468542inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
,:*2Adam/m/conv2d/kernel
,:*2Adam/v/conv2d/kernel
:2Adam/m/conv2d/bias
:2Adam/v/conv2d/bias
,:*2 Adam/m/group_normalization/gamma
,:*2 Adam/v/group_normalization/gamma
+:)2Adam/m/group_normalization/beta
+:)2Adam/v/group_normalization/beta
.:,2Adam/m/conv2d_1/kernel
.:,2Adam/v/conv2d_1/kernel
 :2Adam/m/conv2d_1/bias
 :2Adam/v/conv2d_1/bias
.:,2"Adam/m/group_normalization_1/gamma
.:,2"Adam/v/group_normalization_1/gamma
-:+2!Adam/m/group_normalization_1/beta
-:+2!Adam/v/group_normalization_1/beta
.:,2Adam/m/conv2d_2/kernel
.:,2Adam/v/conv2d_2/kernel
 :2Adam/m/conv2d_2/bias
 :2Adam/v/conv2d_2/bias
.:,2"Adam/m/group_normalization_2/gamma
.:,2"Adam/v/group_normalization_2/gamma
-:+2!Adam/m/group_normalization_2/beta
-:+2!Adam/v/group_normalization_2/beta
.:, 2Adam/m/conv2d_3/kernel
.:, 2Adam/v/conv2d_3/kernel
 : 2Adam/m/conv2d_3/bias
 : 2Adam/v/conv2d_3/bias
.:, 2"Adam/m/group_normalization_3/gamma
.:, 2"Adam/v/group_normalization_3/gamma
-:+ 2!Adam/m/group_normalization_3/beta
-:+ 2!Adam/v/group_normalization_3/beta
.:, @2Adam/m/conv2d_4/kernel
.:, @2Adam/v/conv2d_4/kernel
 :@2Adam/m/conv2d_4/bias
 :@2Adam/v/conv2d_4/bias
.:,@2"Adam/m/group_normalization_4/gamma
.:,@2"Adam/v/group_normalization_4/gamma
-:+@2!Adam/m/group_normalization_4/beta
-:+@2!Adam/v/group_normalization_4/beta
.:,@ 2Adam/m/conv2d_5/kernel
.:,@ 2Adam/v/conv2d_5/kernel
 : 2Adam/m/conv2d_5/bias
 : 2Adam/v/conv2d_5/bias
.:, 2"Adam/m/group_normalization_5/gamma
.:, 2"Adam/v/group_normalization_5/gamma
-:+ 2!Adam/m/group_normalization_5/beta
-:+ 2!Adam/v/group_normalization_5/beta
.:,@2Adam/m/conv2d_6/kernel
.:,@2Adam/v/conv2d_6/kernel
 :2Adam/m/conv2d_6/bias
 :2Adam/v/conv2d_6/bias
.:,2"Adam/m/group_normalization_6/gamma
.:,2"Adam/v/group_normalization_6/gamma
-:+2!Adam/m/group_normalization_6/beta
-:+2!Adam/v/group_normalization_6/beta
.:, 2Adam/m/conv2d_7/kernel
.:, 2Adam/v/conv2d_7/kernel
 :2Adam/m/conv2d_7/bias
 :2Adam/v/conv2d_7/bias
.:,2"Adam/m/group_normalization_7/gamma
.:,2"Adam/v/group_normalization_7/gamma
-:+2!Adam/m/group_normalization_7/beta
-:+2!Adam/v/group_normalization_7/beta
.:,2Adam/m/conv2d_8/kernel
.:,2Adam/v/conv2d_8/kernel
 :2Adam/m/conv2d_8/bias
 :2Adam/v/conv2d_8/bias
.:,2"Adam/m/group_normalization_8/gamma
.:,2"Adam/v/group_normalization_8/gamma
-:+2!Adam/m/group_normalization_8/beta
-:+2!Adam/v/group_normalization_8/beta
.:,2Adam/m/conv2d_9/kernel
.:,2Adam/v/conv2d_9/kernel
 :2Adam/m/conv2d_9/bias
 :2Adam/v/conv2d_9/bias
эBъ
"__inference__update_step_xla_44371gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44376gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44381gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44386gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44391gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44396gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44401gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44406gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44411gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44416gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44421gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44426gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44431gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44436gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44441gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44446gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44451gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44456gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44461gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44466gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44471gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44476gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44481gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44486gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44491gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44496gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44501gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44506gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44511gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44516gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44521gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44526gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44531gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44536gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44541gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44546gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44551gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_44556gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2countЄ
"__inference__update_step_xla_44371~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`РдрРћ?
Њ "
 
"__inference__update_step_xla_44376f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` пЭРћ?
Њ "
 
"__inference__update_step_xla_44381f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` Сћ?
Њ "
 
"__inference__update_step_xla_44386f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` Сћ?
Њ "
 Є
"__inference__update_step_xla_44391~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
` тЬєњ?
Њ "
 
"__inference__update_step_xla_44396f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ьЬєњ?
Њ "
 
"__inference__update_step_xla_44401f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` цнПћ?
Њ "
 
"__inference__update_step_xla_44406f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ЫЭєњ?
Њ "
 Є
"__inference__update_step_xla_44411~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`Рбеєњ?
Њ "
 
"__inference__update_step_xla_44416f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`Ќеєњ?
Њ "
 
"__inference__update_step_xla_44421f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рХжєњ?
Њ "
 
"__inference__update_step_xla_44426f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`Дкєњ?
Њ "
 Є
"__inference__update_step_xla_44431~xЂu
nЂk
!
gradient 
<9	%Ђ"
њ 

p
` VariableSpec 
` Ётєњ?
Њ "
 
"__inference__update_step_xla_44436f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`рМтєњ?
Њ "
 
"__inference__update_step_xla_44441f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`рзєњ?
Њ "
 
"__inference__update_step_xla_44446f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`Їуєњ?
Њ "
 Є
"__inference__update_step_xla_44451~xЂu
nЂk
!
gradient @
<9	%Ђ"
њ @

p
` VariableSpec 
` Ылєњ?
Њ "
 
"__inference__update_step_xla_44456f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рЅлєњ?
Њ "
 
"__inference__update_step_xla_44461f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рснПћ?
Њ "
 
"__inference__update_step_xla_44466f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` гкєњ?
Њ "
 Є
"__inference__update_step_xla_44471~xЂu
nЂk
!
gradient@ 
<9	%Ђ"
њ@ 

p
` VariableSpec 
`ДЬєњ?
Њ "
 
"__inference__update_step_xla_44476f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`РдЬєњ?
Њ "
 
"__inference__update_step_xla_44481f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`рзєњ?
Њ "
 
"__inference__update_step_xla_44486f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
` ЋЄСћ?
Њ "
 Є
"__inference__update_step_xla_44491~xЂu
nЂk
!
gradient@
<9	%Ђ"
њ@

p
` VariableSpec 
`Рєэєњ?
Њ "
 
"__inference__update_step_xla_44496f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`Рбэєњ?
Њ "
 
"__inference__update_step_xla_44501f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`  жєњ?
Њ "
 
"__inference__update_step_xla_44506f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`Рњєњ?
Њ "
 Є
"__inference__update_step_xla_44511~xЂu
nЂk
!
gradient 
<9	%Ђ"
њ 

p
` VariableSpec 
`єњ?
Њ "
 
"__inference__update_step_xla_44516f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`єњ?
Њ "
 
"__inference__update_step_xla_44521f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` Ієњ?
Њ "
 
"__inference__update_step_xla_44526f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рСєњ?
Њ "
 Є
"__inference__update_step_xla_44531~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`єњ?
Њ "
 
"__inference__update_step_xla_44536f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` єњ?
Њ "
 
"__inference__update_step_xla_44541f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`різєњ?
Њ "
 
"__inference__update_step_xla_44546f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` Зєњ?
Њ "
 Є
"__inference__update_step_xla_44551~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`рЫєњ?
Њ "
 
"__inference__update_step_xla_44556f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рзєњ?
Њ "
 
"__inference__wrapped_model_6465732пB;<JKXYghuvЁЂЏАОПвглмѕіўџЁЂЛМФХопJЂG
@Ђ=
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "MЊJ
H
conv2d_9<9
conv2d_9+џџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_6467527ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_average_pooling2d_1_layer_call_fn_6467522RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_6467643ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_average_pooling2d_2_layer_call_fn_6467638RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_6467759ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_average_pooling2d_3_layer_call_fn_6467754RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_6467875ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_average_pooling2d_4_layer_call_fn_6467870RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџј
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_6467411ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 в
3__inference_average_pooling2d_layer_call_fn_6467406RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЋ
J__inference_concatenate_1_layer_call_and_return_conditional_losses_6468234мЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
/__inference_concatenate_1_layer_call_fn_6468227бЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ћ
J__inference_concatenate_2_layer_call_and_return_conditional_losses_6468370мЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
/__inference_concatenate_2_layer_call_fn_6468363бЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЋ
J__inference_concatenate_3_layer_call_and_return_conditional_losses_6468506мЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
/__inference_concatenate_3_layer_call_fn_6468499бЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
H__inference_concatenate_layer_call_and_return_conditional_losses_6468098мЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
-__inference_concatenate_layer_call_fn_6468091бЂ
Ђ
|
<9
inputs_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
<9
inputs_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@с
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467517XYIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Л
*__inference_conv2d_1_layer_call_fn_6467507XYIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџс
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6467633uvIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Л
*__inference_conv2d_2_layer_call_fn_6467623uvIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџу
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6467749IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Н
*__inference_conv2d_3_layer_call_fn_6467739IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ у
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6467865ЏАIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Н
*__inference_conv2d_4_layer_call_fn_6467855ЏАIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@у
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6467998вгIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Н
*__inference_conv2d_5_layer_call_fn_6467988вгIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ у
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6468134ѕіIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
*__inference_conv2d_6_layer_call_fn_6468124ѕіIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџу
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6468270IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
*__inference_conv2d_7_layer_call_fn_6468260IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџу
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6468406ЛМIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
*__inference_conv2d_8_layer_call_fn_6468396ЛМIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџу
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6468542опIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
*__inference_conv2d_9_layer_call_fn_6468532опIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџп
C__inference_conv2d_layer_call_and_return_conditional_losses_6467401;<IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Й
(__inference_conv2d_layer_call_fn_6467391;<IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџю
R__inference_group_normalization_1_layer_call_and_return_conditional_losses_6467604ghIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
7__inference_group_normalization_1_layer_call_fn_6467536ghIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ№
R__inference_group_normalization_2_layer_call_and_return_conditional_losses_6467720IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ъ
7__inference_group_normalization_2_layer_call_fn_6467652IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ№
R__inference_group_normalization_3_layer_call_and_return_conditional_losses_6467836ЁЂIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ъ
7__inference_group_normalization_3_layer_call_fn_6467768ЁЂIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ №
R__inference_group_normalization_4_layer_call_and_return_conditional_losses_6467952ОПIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ъ
7__inference_group_normalization_4_layer_call_fn_6467884ОПIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@№
R__inference_group_normalization_5_layer_call_and_return_conditional_losses_6468075лмIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ъ
7__inference_group_normalization_5_layer_call_fn_6468007лмIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ №
R__inference_group_normalization_6_layer_call_and_return_conditional_losses_6468211ўџIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ъ
7__inference_group_normalization_6_layer_call_fn_6468143ўџIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ№
R__inference_group_normalization_7_layer_call_and_return_conditional_losses_6468347ЁЂIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ъ
7__inference_group_normalization_7_layer_call_fn_6468279ЁЂIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ№
R__inference_group_normalization_8_layer_call_and_return_conditional_losses_6468483ФХIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ъ
7__inference_group_normalization_8_layer_call_fn_6468415ФХIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџь
P__inference_group_normalization_layer_call_and_return_conditional_losses_6467488JKIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
5__inference_group_normalization_layer_call_fn_6467420JKIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЇ
B__inference_model_layer_call_and_return_conditional_losses_6466855рB;<JKXYghuvЁЂЏАОПвглмѕіўџЁЂЛМФХопRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ї
B__inference_model_layer_call_and_return_conditional_losses_6466977рB;<JKXYghuvЁЂЏАОПвглмѕіўџЁЂЛМФХопRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
'__inference_model_layer_call_fn_6467058еB;<JKXYghuvЁЂЏАОПвглмѕіўџЁЂЛМФХопRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
'__inference_model_layer_call_fn_6467139еB;<JKXYghuvЁЂЏАОПвглмѕіўџЁЂЛМФХопRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџм
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6467614IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
)__inference_re_lu_1_layer_call_fn_6467609IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџм
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6467730IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
)__inference_re_lu_2_layer_call_fn_6467725IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџм
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6467846IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ж
)__inference_re_lu_3_layer_call_fn_6467841IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ м
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6467962IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ж
)__inference_re_lu_4_layer_call_fn_6467957IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@м
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6468085IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ж
)__inference_re_lu_5_layer_call_fn_6468080IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ м
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6468221IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
)__inference_re_lu_6_layer_call_fn_6468216IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџм
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6468357IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
)__inference_re_lu_7_layer_call_fn_6468352IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџм
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6468493IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
)__inference_re_lu_8_layer_call_fn_6468488IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџк
B__inference_re_lu_layer_call_and_return_conditional_losses_6467498IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
'__inference_re_lu_layer_call_fn_6467493IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
%__inference_signature_wrapper_6467382ъB;<JKXYghuvЁЂЏАОПвглмѕіўџЁЂЛМФХопUЂR
Ђ 
KЊH
F
input_1;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"MЊJ
H
conv2d_9<9
conv2d_9+џџџџџџџџџџџџџџџџџџџџџџџџџџџі
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6468115ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_up_sampling2d_1_layer_call_fn_6468103RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџі
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6468251ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_up_sampling2d_2_layer_call_fn_6468239RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџі
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6468387ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_up_sampling2d_3_layer_call_fn_6468375RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџі
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6468523ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_up_sampling2d_4_layer_call_fn_6468511RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6467979ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_up_sampling2d_layer_call_fn_6467967RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ