┌╫
ё─
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8ЭЪ
В
Adam/dense_182/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_182/bias/v
{
)Adam/dense_182/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_182/bias/v*
_output_shapes
:/*
dtype0
К
Adam/dense_182/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@/*(
shared_nameAdam/dense_182/kernel/v
Г
+Adam/dense_182/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_182/kernel/v*
_output_shapes

:@/*
dtype0
В
Adam/dense_181/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_181/bias/v
{
)Adam/dense_181/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_181/bias/v*
_output_shapes
:@*
dtype0
К
Adam/dense_181/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_181/kernel/v
Г
+Adam/dense_181/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_181/kernel/v*
_output_shapes

:@@*
dtype0
В
Adam/dense_180/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_180/bias/v
{
)Adam/dense_180/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_180/bias/v*
_output_shapes
:@*
dtype0
Л
Adam/dense_180/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*(
shared_nameAdam/dense_180/kernel/v
Д
+Adam/dense_180/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_180/kernel/v*
_output_shapes
:	А@*
dtype0
Е
Adam/conv2d_182/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/conv2d_182/bias/v
~
*Adam/conv2d_182/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_182/bias/v*
_output_shapes	
:А*
dtype0
Х
Adam/conv2d_182/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*)
shared_nameAdam/conv2d_182/kernel/v
О
,Adam/conv2d_182/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_182/kernel/v*'
_output_shapes
:@А*
dtype0
Д
Adam/conv2d_181/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_181/bias/v
}
*Adam/conv2d_181/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_181/bias/v*
_output_shapes
:@*
dtype0
Ф
Adam/conv2d_181/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_181/kernel/v
Н
,Adam/conv2d_181/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_181/kernel/v*&
_output_shapes
: @*
dtype0
Д
Adam/conv2d_180/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_180/bias/v
}
*Adam/conv2d_180/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_180/bias/v*
_output_shapes
: *
dtype0
Ф
Adam/conv2d_180/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_180/kernel/v
Н
,Adam/conv2d_180/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_180/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/dense_182/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_182/bias/m
{
)Adam/dense_182/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_182/bias/m*
_output_shapes
:/*
dtype0
К
Adam/dense_182/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@/*(
shared_nameAdam/dense_182/kernel/m
Г
+Adam/dense_182/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_182/kernel/m*
_output_shapes

:@/*
dtype0
В
Adam/dense_181/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_181/bias/m
{
)Adam/dense_181/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_181/bias/m*
_output_shapes
:@*
dtype0
К
Adam/dense_181/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_181/kernel/m
Г
+Adam/dense_181/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_181/kernel/m*
_output_shapes

:@@*
dtype0
В
Adam/dense_180/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_180/bias/m
{
)Adam/dense_180/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_180/bias/m*
_output_shapes
:@*
dtype0
Л
Adam/dense_180/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*(
shared_nameAdam/dense_180/kernel/m
Д
+Adam/dense_180/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_180/kernel/m*
_output_shapes
:	А@*
dtype0
Е
Adam/conv2d_182/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/conv2d_182/bias/m
~
*Adam/conv2d_182/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_182/bias/m*
_output_shapes	
:А*
dtype0
Х
Adam/conv2d_182/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*)
shared_nameAdam/conv2d_182/kernel/m
О
,Adam/conv2d_182/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_182/kernel/m*'
_output_shapes
:@А*
dtype0
Д
Adam/conv2d_181/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_181/bias/m
}
*Adam/conv2d_181/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_181/bias/m*
_output_shapes
:@*
dtype0
Ф
Adam/conv2d_181/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_181/kernel/m
Н
,Adam/conv2d_181/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_181/kernel/m*&
_output_shapes
: @*
dtype0
Д
Adam/conv2d_180/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_180/bias/m
}
*Adam/conv2d_180/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_180/bias/m*
_output_shapes
: *
dtype0
Ф
Adam/conv2d_180/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_180/kernel/m
Н
,Adam/conv2d_180/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_180/kernel/m*&
_output_shapes
: *
dtype0
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
t
dense_182/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_182/bias
m
"dense_182/bias/Read/ReadVariableOpReadVariableOpdense_182/bias*
_output_shapes
:/*
dtype0
|
dense_182/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@/*!
shared_namedense_182/kernel
u
$dense_182/kernel/Read/ReadVariableOpReadVariableOpdense_182/kernel*
_output_shapes

:@/*
dtype0
t
dense_181/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_181/bias
m
"dense_181/bias/Read/ReadVariableOpReadVariableOpdense_181/bias*
_output_shapes
:@*
dtype0
|
dense_181/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_181/kernel
u
$dense_181/kernel/Read/ReadVariableOpReadVariableOpdense_181/kernel*
_output_shapes

:@@*
dtype0
t
dense_180/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_180/bias
m
"dense_180/bias/Read/ReadVariableOpReadVariableOpdense_180/bias*
_output_shapes
:@*
dtype0
}
dense_180/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*!
shared_namedense_180/kernel
v
$dense_180/kernel/Read/ReadVariableOpReadVariableOpdense_180/kernel*
_output_shapes
:	А@*
dtype0
w
conv2d_182/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_182/bias
p
#conv2d_182/bias/Read/ReadVariableOpReadVariableOpconv2d_182/bias*
_output_shapes	
:А*
dtype0
З
conv2d_182/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_182/kernel
А
%conv2d_182/kernel/Read/ReadVariableOpReadVariableOpconv2d_182/kernel*'
_output_shapes
:@А*
dtype0
v
conv2d_181/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_181/bias
o
#conv2d_181/bias/Read/ReadVariableOpReadVariableOpconv2d_181/bias*
_output_shapes
:@*
dtype0
Ж
conv2d_181/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_181/kernel

%conv2d_181/kernel/Read/ReadVariableOpReadVariableOpconv2d_181/kernel*&
_output_shapes
: @*
dtype0
v
conv2d_180/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_180/bias
o
#conv2d_180/bias/Read/ReadVariableOpReadVariableOpconv2d_180/bias*
_output_shapes
: *
dtype0
Ж
conv2d_180/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_180/kernel

%conv2d_180/kernel/Read/ReadVariableOpReadVariableOpconv2d_180/kernel*&
_output_shapes
: *
dtype0
Л
serving_default_input_61Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
в
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_61conv2d_180/kernelconv2d_180/biasconv2d_181/kernelconv2d_181/biasconv2d_182/kernelconv2d_182/biasdense_180/kerneldense_180/biasdense_181/kerneldense_181/biasdense_182/kerneldense_182/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_25074123

NoOpNoOp
еh
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*рg
value╓gB╙g B╠g
°
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
╚
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op*
О
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
╚
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op*
О
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
ж
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
е
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator* 
ж
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias*
е
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator* 
ж
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias*
Z
0
1
*2
+3
94
:5
H6
I7
W8
X9
f10
g11*
Z
0
1
*2
+3
94
:5
H6
I7
W8
X9
f10
g11*
%
h0
i1
j2
k3
l4* 
░
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
rtrace_0
strace_1
ttrace_2
utrace_3* 
6
vtrace_0
wtrace_1
xtrace_2
ytrace_3* 
* 
┤
ziter

{beta_1

|beta_2
	}decay
~learning_ratemсmт*mу+mф9mх:mцHmчImшWmщXmъfmыgmьvэvю*vя+vЁ9vё:vЄHvєIvЇWvїXvЎfvўgv°*

serving_default* 

0
1*

0
1*
	
h0* 
Ш
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
a[
VARIABLE_VALUEconv2d_180/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_180/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

Мtrace_0* 

Нtrace_0* 

*0
+1*

*0
+1*
	
i0* 
Ш
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
a[
VARIABLE_VALUEconv2d_181/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_181/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 

90
:1*

90
:1*
	
j0* 
Ш
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
a[
VARIABLE_VALUEconv2d_182/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_182/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

иtrace_0* 

йtrace_0* 

H0
I1*

H0
I1*
	
k0* 
Ш
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

пtrace_0* 

░trace_0* 
`Z
VARIABLE_VALUEdense_180/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_180/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

╢trace_0
╖trace_1* 

╕trace_0
╣trace_1* 
* 

W0
X1*

W0
X1*
	
l0* 
Ш
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

┐trace_0* 

└trace_0* 
`Z
VARIABLE_VALUEdense_181/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_181/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

╞trace_0
╟trace_1* 

╚trace_0
╔trace_1* 
* 

f0
g1*

f0
g1*
* 
Ш
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

╧trace_0* 

╨trace_0* 
`Z
VARIABLE_VALUEdense_182/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_182/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

╤trace_0* 

╥trace_0* 

╙trace_0* 

╘trace_0* 

╒trace_0* 
* 
R
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
10*

╓0
╫1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
h0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
i0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
j0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
k0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
l0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
╪	variables
┘	keras_api

┌total

█count*
M
▄	variables
▌	keras_api

▐total

▀count
р
_fn_kwargs*

┌0
█1*

╪	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

▐0
▀1*

▄	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Д~
VARIABLE_VALUEAdam/conv2d_180/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_180/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_181/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_181/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_182/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_182/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_180/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_180/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_181/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_181/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_182/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_182/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_180/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_180/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_181/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_181/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_182/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_182/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_180/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_180/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_181/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_181/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_182/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_182/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╘
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_180/kernel/Read/ReadVariableOp#conv2d_180/bias/Read/ReadVariableOp%conv2d_181/kernel/Read/ReadVariableOp#conv2d_181/bias/Read/ReadVariableOp%conv2d_182/kernel/Read/ReadVariableOp#conv2d_182/bias/Read/ReadVariableOp$dense_180/kernel/Read/ReadVariableOp"dense_180/bias/Read/ReadVariableOp$dense_181/kernel/Read/ReadVariableOp"dense_181/bias/Read/ReadVariableOp$dense_182/kernel/Read/ReadVariableOp"dense_182/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_180/kernel/m/Read/ReadVariableOp*Adam/conv2d_180/bias/m/Read/ReadVariableOp,Adam/conv2d_181/kernel/m/Read/ReadVariableOp*Adam/conv2d_181/bias/m/Read/ReadVariableOp,Adam/conv2d_182/kernel/m/Read/ReadVariableOp*Adam/conv2d_182/bias/m/Read/ReadVariableOp+Adam/dense_180/kernel/m/Read/ReadVariableOp)Adam/dense_180/bias/m/Read/ReadVariableOp+Adam/dense_181/kernel/m/Read/ReadVariableOp)Adam/dense_181/bias/m/Read/ReadVariableOp+Adam/dense_182/kernel/m/Read/ReadVariableOp)Adam/dense_182/bias/m/Read/ReadVariableOp,Adam/conv2d_180/kernel/v/Read/ReadVariableOp*Adam/conv2d_180/bias/v/Read/ReadVariableOp,Adam/conv2d_181/kernel/v/Read/ReadVariableOp*Adam/conv2d_181/bias/v/Read/ReadVariableOp,Adam/conv2d_182/kernel/v/Read/ReadVariableOp*Adam/conv2d_182/bias/v/Read/ReadVariableOp+Adam/dense_180/kernel/v/Read/ReadVariableOp)Adam/dense_180/bias/v/Read/ReadVariableOp+Adam/dense_181/kernel/v/Read/ReadVariableOp)Adam/dense_181/bias/v/Read/ReadVariableOp+Adam/dense_182/kernel/v/Read/ReadVariableOp)Adam/dense_182/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_save_25074784
╦	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_180/kernelconv2d_180/biasconv2d_181/kernelconv2d_181/biasconv2d_182/kernelconv2d_182/biasdense_180/kerneldense_180/biasdense_181/kerneldense_181/biasdense_182/kerneldense_182/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_180/kernel/mAdam/conv2d_180/bias/mAdam/conv2d_181/kernel/mAdam/conv2d_181/bias/mAdam/conv2d_182/kernel/mAdam/conv2d_182/bias/mAdam/dense_180/kernel/mAdam/dense_180/bias/mAdam/dense_181/kernel/mAdam/dense_181/bias/mAdam/dense_182/kernel/mAdam/dense_182/bias/mAdam/conv2d_180/kernel/vAdam/conv2d_180/bias/vAdam/conv2d_181/kernel/vAdam/conv2d_181/bias/vAdam/conv2d_182/kernel/vAdam/conv2d_182/bias/vAdam/dense_180/kernel/vAdam/dense_180/bias/vAdam/dense_181/kernel/vAdam/dense_181/bias/vAdam/dense_182/kernel/vAdam/dense_182/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference__traced_restore_25074929Ф╣

╨	
└
__inference_loss_fn_2_25074608W
<conv2d_182_kernel_regularizer_l2loss_readvariableop_resource:@А
identityИв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp╣
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<conv2d_182_kernel_regularizer_l2loss_readvariableop_resource*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_182/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp
°

┐
&__inference_signature_wrapper_25074123
input_61!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@/

unknown_10:/
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_25073466o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_61
█
f
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074499

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
єl
ї
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074357

inputsC
)conv2d_180_conv2d_readvariableop_resource: 8
*conv2d_180_biasadd_readvariableop_resource: C
)conv2d_181_conv2d_readvariableop_resource: @8
*conv2d_181_biasadd_readvariableop_resource:@D
)conv2d_182_conv2d_readvariableop_resource:@А9
*conv2d_182_biasadd_readvariableop_resource:	А;
(dense_180_matmul_readvariableop_resource:	А@7
)dense_180_biasadd_readvariableop_resource:@:
(dense_181_matmul_readvariableop_resource:@@7
)dense_181_biasadd_readvariableop_resource:@:
(dense_182_matmul_readvariableop_resource:@/7
)dense_182_biasadd_readvariableop_resource:/
identityИв!conv2d_180/BiasAdd/ReadVariableOpв conv2d_180/Conv2D/ReadVariableOpв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_181/BiasAdd/ReadVariableOpв conv2d_181/Conv2D/ReadVariableOpв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_182/BiasAdd/ReadVariableOpв conv2d_182/Conv2D/ReadVariableOpв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpв dense_180/BiasAdd/ReadVariableOpвdense_180/MatMul/ReadVariableOpв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpв dense_181/BiasAdd/ReadVariableOpвdense_181/MatMul/ReadVariableOpв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpв dense_182/BiasAdd/ReadVariableOpвdense_182/MatMul/ReadVariableOpТ
 conv2d_180/Conv2D/ReadVariableOpReadVariableOp)conv2d_180_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0п
conv2d_180/Conv2DConv2Dinputs(conv2d_180/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
И
!conv2d_180/BiasAdd/ReadVariableOpReadVariableOp*conv2d_180_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
conv2d_180/BiasAddBiasAddconv2d_180/Conv2D:output:0)conv2d_180/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          n
conv2d_180/ReluReluconv2d_180/BiasAdd:output:0*
T0*/
_output_shapes
:          ░
max_pooling2d_120/MaxPoolMaxPoolconv2d_180/Relu:activations:0*/
_output_shapes
:         		 *
ksize
*
paddingVALID*
strides
Т
 conv2d_181/Conv2D/ReadVariableOpReadVariableOp)conv2d_181_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╦
conv2d_181/Conv2DConv2D"max_pooling2d_120/MaxPool:output:0(conv2d_181/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingSAME*
strides
И
!conv2d_181/BiasAdd/ReadVariableOpReadVariableOp*conv2d_181_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
conv2d_181/BiasAddBiasAddconv2d_181/Conv2D:output:0)conv2d_181/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@n
conv2d_181/ReluReluconv2d_181/BiasAdd:output:0*
T0*/
_output_shapes
:         		@░
max_pooling2d_121/MaxPoolMaxPoolconv2d_181/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
У
 conv2d_182/Conv2D/ReadVariableOpReadVariableOp)conv2d_182_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0═
conv2d_182/Conv2DConv2D"max_pooling2d_121/MaxPool:output:0(conv2d_182/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
Й
!conv2d_182/BiasAdd/ReadVariableOpReadVariableOp*conv2d_182_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Я
conv2d_182/BiasAddBiasAddconv2d_182/Conv2D:output:0)conv2d_182/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аo
conv2d_182/ReluReluconv2d_182/BiasAdd:output:0*
T0*0
_output_shapes
:         Аa
flatten_60/ConstConst*
_output_shapes
:*
dtype0*
valueB"       К
flatten_60/ReshapeReshapeconv2d_182/Relu:activations:0flatten_60/Const:output:0*
T0*(
_output_shapes
:         АЙ
dense_180/MatMul/ReadVariableOpReadVariableOp(dense_180_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0Т
dense_180/MatMulMatMulflatten_60/Reshape:output:0'dense_180/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ж
 dense_180/BiasAdd/ReadVariableOpReadVariableOp)dense_180_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
dense_180/BiasAddBiasAdddense_180/MatMul:product:0(dense_180/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
dense_180/ReluReludense_180/BiasAdd:output:0*
T0*'
_output_shapes
:         @]
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Р
dropout_37/dropout/MulMuldense_180/Relu:activations:0!dropout_37/dropout/Const:output:0*
T0*'
_output_shapes
:         @d
dropout_37/dropout/ShapeShapedense_180/Relu:activations:0*
T0*
_output_shapes
:о
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*

seed*f
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╟
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @Е
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @К
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*'
_output_shapes
:         @И
dense_181/MatMul/ReadVariableOpReadVariableOp(dense_181_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0У
dense_181/MatMulMatMuldropout_37/dropout/Mul_1:z:0'dense_181/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ж
 dense_181/BiasAdd/ReadVariableOpReadVariableOp)dense_181_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
dense_181/BiasAddBiasAdddense_181/MatMul:product:0(dense_181/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
dense_181/ReluReludense_181/BiasAdd:output:0*
T0*'
_output_shapes
:         @]
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Р
dropout_38/dropout/MulMuldense_181/Relu:activations:0!dropout_38/dropout/Const:output:0*
T0*'
_output_shapes
:         @d
dropout_38/dropout/ShapeShapedense_181/Relu:activations:0*
T0*
_output_shapes
:╗
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*

seed**
seed2f
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╟
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @Е
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @К
dropout_38/dropout/Mul_1Muldropout_38/dropout/Mul:z:0dropout_38/dropout/Cast:y:0*
T0*'
_output_shapes
:         @И
dense_182/MatMul/ReadVariableOpReadVariableOp(dense_182_matmul_readvariableop_resource*
_output_shapes

:@/*
dtype0У
dense_182/MatMulMatMuldropout_38/dropout/Mul_1:z:0'dense_182/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /Ж
 dense_182/BiasAdd/ReadVariableOpReadVariableOp)dense_182_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0Ф
dense_182/BiasAddBiasAdddense_182/MatMul:product:0(dense_182/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /е
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)conv2d_180_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: е
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)conv2d_181_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ж
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)conv2d_182_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ь
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_180_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_181_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_182/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         /Ў
NoOpNoOp"^conv2d_180/BiasAdd/ReadVariableOp!^conv2d_180/Conv2D/ReadVariableOp4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_181/BiasAdd/ReadVariableOp!^conv2d_181/Conv2D/ReadVariableOp4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_182/BiasAdd/ReadVariableOp!^conv2d_182/Conv2D/ReadVariableOp4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_180/BiasAdd/ReadVariableOp ^dense_180/MatMul/ReadVariableOp3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_181/BiasAdd/ReadVariableOp ^dense_181/MatMul/ReadVariableOp3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_182/BiasAdd/ReadVariableOp ^dense_182/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2F
!conv2d_180/BiasAdd/ReadVariableOp!conv2d_180/BiasAdd/ReadVariableOp2D
 conv2d_180/Conv2D/ReadVariableOp conv2d_180/Conv2D/ReadVariableOp2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_181/BiasAdd/ReadVariableOp!conv2d_181/BiasAdd/ReadVariableOp2D
 conv2d_181/Conv2D/ReadVariableOp conv2d_181/Conv2D/ReadVariableOp2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_182/BiasAdd/ReadVariableOp!conv2d_182/BiasAdd/ReadVariableOp2D
 conv2d_182/Conv2D/ReadVariableOp conv2d_182/Conv2D/ReadVariableOp2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_180/BiasAdd/ReadVariableOp dense_180/BiasAdd/ReadVariableOp2B
dense_180/MatMul/ReadVariableOpdense_180/MatMul/ReadVariableOp2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_181/BiasAdd/ReadVariableOp dense_181/BiasAdd/ReadVariableOp2B
dense_181/MatMul/ReadVariableOpdense_181/MatMul/ReadVariableOp2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_182/BiasAdd/ReadVariableOp dense_182/BiasAdd/ReadVariableOp2B
dense_182/MatMul/ReadVariableOpdense_182/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
и]
ї
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074272

inputsC
)conv2d_180_conv2d_readvariableop_resource: 8
*conv2d_180_biasadd_readvariableop_resource: C
)conv2d_181_conv2d_readvariableop_resource: @8
*conv2d_181_biasadd_readvariableop_resource:@D
)conv2d_182_conv2d_readvariableop_resource:@А9
*conv2d_182_biasadd_readvariableop_resource:	А;
(dense_180_matmul_readvariableop_resource:	А@7
)dense_180_biasadd_readvariableop_resource:@:
(dense_181_matmul_readvariableop_resource:@@7
)dense_181_biasadd_readvariableop_resource:@:
(dense_182_matmul_readvariableop_resource:@/7
)dense_182_biasadd_readvariableop_resource:/
identityИв!conv2d_180/BiasAdd/ReadVariableOpв conv2d_180/Conv2D/ReadVariableOpв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_181/BiasAdd/ReadVariableOpв conv2d_181/Conv2D/ReadVariableOpв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_182/BiasAdd/ReadVariableOpв conv2d_182/Conv2D/ReadVariableOpв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpв dense_180/BiasAdd/ReadVariableOpвdense_180/MatMul/ReadVariableOpв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpв dense_181/BiasAdd/ReadVariableOpвdense_181/MatMul/ReadVariableOpв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpв dense_182/BiasAdd/ReadVariableOpвdense_182/MatMul/ReadVariableOpТ
 conv2d_180/Conv2D/ReadVariableOpReadVariableOp)conv2d_180_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0п
conv2d_180/Conv2DConv2Dinputs(conv2d_180/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
И
!conv2d_180/BiasAdd/ReadVariableOpReadVariableOp*conv2d_180_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
conv2d_180/BiasAddBiasAddconv2d_180/Conv2D:output:0)conv2d_180/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          n
conv2d_180/ReluReluconv2d_180/BiasAdd:output:0*
T0*/
_output_shapes
:          ░
max_pooling2d_120/MaxPoolMaxPoolconv2d_180/Relu:activations:0*/
_output_shapes
:         		 *
ksize
*
paddingVALID*
strides
Т
 conv2d_181/Conv2D/ReadVariableOpReadVariableOp)conv2d_181_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╦
conv2d_181/Conv2DConv2D"max_pooling2d_120/MaxPool:output:0(conv2d_181/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingSAME*
strides
И
!conv2d_181/BiasAdd/ReadVariableOpReadVariableOp*conv2d_181_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
conv2d_181/BiasAddBiasAddconv2d_181/Conv2D:output:0)conv2d_181/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@n
conv2d_181/ReluReluconv2d_181/BiasAdd:output:0*
T0*/
_output_shapes
:         		@░
max_pooling2d_121/MaxPoolMaxPoolconv2d_181/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
У
 conv2d_182/Conv2D/ReadVariableOpReadVariableOp)conv2d_182_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0═
conv2d_182/Conv2DConv2D"max_pooling2d_121/MaxPool:output:0(conv2d_182/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
Й
!conv2d_182/BiasAdd/ReadVariableOpReadVariableOp*conv2d_182_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Я
conv2d_182/BiasAddBiasAddconv2d_182/Conv2D:output:0)conv2d_182/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аo
conv2d_182/ReluReluconv2d_182/BiasAdd:output:0*
T0*0
_output_shapes
:         Аa
flatten_60/ConstConst*
_output_shapes
:*
dtype0*
valueB"       К
flatten_60/ReshapeReshapeconv2d_182/Relu:activations:0flatten_60/Const:output:0*
T0*(
_output_shapes
:         АЙ
dense_180/MatMul/ReadVariableOpReadVariableOp(dense_180_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0Т
dense_180/MatMulMatMulflatten_60/Reshape:output:0'dense_180/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ж
 dense_180/BiasAdd/ReadVariableOpReadVariableOp)dense_180_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
dense_180/BiasAddBiasAdddense_180/MatMul:product:0(dense_180/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
dense_180/ReluReludense_180/BiasAdd:output:0*
T0*'
_output_shapes
:         @o
dropout_37/IdentityIdentitydense_180/Relu:activations:0*
T0*'
_output_shapes
:         @И
dense_181/MatMul/ReadVariableOpReadVariableOp(dense_181_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0У
dense_181/MatMulMatMuldropout_37/Identity:output:0'dense_181/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ж
 dense_181/BiasAdd/ReadVariableOpReadVariableOp)dense_181_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
dense_181/BiasAddBiasAdddense_181/MatMul:product:0(dense_181/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
dense_181/ReluReludense_181/BiasAdd:output:0*
T0*'
_output_shapes
:         @o
dropout_38/IdentityIdentitydense_181/Relu:activations:0*
T0*'
_output_shapes
:         @И
dense_182/MatMul/ReadVariableOpReadVariableOp(dense_182_matmul_readvariableop_resource*
_output_shapes

:@/*
dtype0У
dense_182/MatMulMatMuldropout_38/Identity:output:0'dense_182/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /Ж
 dense_182/BiasAdd/ReadVariableOpReadVariableOp)dense_182_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0Ф
dense_182/BiasAddBiasAdddense_182/MatMul:product:0(dense_182/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /е
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)conv2d_180_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: е
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)conv2d_181_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ж
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp)conv2d_182_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ь
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_180_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_181_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_182/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         /Ў
NoOpNoOp"^conv2d_180/BiasAdd/ReadVariableOp!^conv2d_180/Conv2D/ReadVariableOp4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_181/BiasAdd/ReadVariableOp!^conv2d_181/Conv2D/ReadVariableOp4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_182/BiasAdd/ReadVariableOp!^conv2d_182/Conv2D/ReadVariableOp4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_180/BiasAdd/ReadVariableOp ^dense_180/MatMul/ReadVariableOp3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_181/BiasAdd/ReadVariableOp ^dense_181/MatMul/ReadVariableOp3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_182/BiasAdd/ReadVariableOp ^dense_182/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2F
!conv2d_180/BiasAdd/ReadVariableOp!conv2d_180/BiasAdd/ReadVariableOp2D
 conv2d_180/Conv2D/ReadVariableOp conv2d_180/Conv2D/ReadVariableOp2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_181/BiasAdd/ReadVariableOp!conv2d_181/BiasAdd/ReadVariableOp2D
 conv2d_181/Conv2D/ReadVariableOp conv2d_181/Conv2D/ReadVariableOp2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_182/BiasAdd/ReadVariableOp!conv2d_182/BiasAdd/ReadVariableOp2D
 conv2d_182/Conv2D/ReadVariableOp conv2d_182/Conv2D/ReadVariableOp2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_180/BiasAdd/ReadVariableOp dense_180/BiasAdd/ReadVariableOp2B
dense_180/MatMul/ReadVariableOpdense_180/MatMul/ReadVariableOp2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_181/BiasAdd/ReadVariableOp dense_181/BiasAdd/ReadVariableOp2B
dense_181/MatMul/ReadVariableOpdense_181/MatMul/ReadVariableOp2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_182/BiasAdd/ReadVariableOp dense_182/BiasAdd/ReadVariableOp2B
dense_182/MatMul/ReadVariableOpdense_182/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╩	
°
G__inference_dense_182_layer_call_and_return_conditional_losses_25074581

inputs0
matmul_readvariableop_resource:@/-
biasadd_readvariableop_resource:/
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         /w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╠
d
H__inference_flatten_60_layer_call_and_return_conditional_losses_25074460

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
уQ
ш
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25073892

inputs-
conv2d_180_25073836: !
conv2d_180_25073838: -
conv2d_181_25073842: @!
conv2d_181_25073844:@.
conv2d_182_25073848:@А"
conv2d_182_25073850:	А%
dense_180_25073854:	А@ 
dense_180_25073856:@$
dense_181_25073860:@@ 
dense_181_25073862:@$
dense_182_25073866:@/ 
dense_182_25073868:/
identityИв"conv2d_180/StatefulPartitionedCallв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_181/StatefulPartitionedCallв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_182/StatefulPartitionedCallв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_180/StatefulPartitionedCallв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_181/StatefulPartitionedCallв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_182/StatefulPartitionedCallв"dropout_37/StatefulPartitionedCallв"dropout_38/StatefulPartitionedCallЙ
"conv2d_180/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_180_25073836conv2d_180_25073838*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25073512№
!max_pooling2d_120/PartitionedCallPartitionedCall+conv2d_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25073475н
"conv2d_181/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_120/PartitionedCall:output:0conv2d_181_25073842conv2d_181_25073844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25073534№
!max_pooling2d_121/PartitionedCallPartitionedCall+conv2d_181/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25073487о
"conv2d_182/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_121/PartitionedCall:output:0conv2d_182_25073848conv2d_182_25073850*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25073556ч
flatten_60/PartitionedCallPartitionedCall+conv2d_182/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_60_layer_call_and_return_conditional_losses_25073568Ъ
!dense_180/StatefulPartitionedCallStatefulPartitionedCall#flatten_60/PartitionedCall:output:0dense_180_25073854dense_180_25073856*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_180_layer_call_and_return_conditional_losses_25073585ї
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall*dense_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073753в
!dense_181/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_181_25073860dense_181_25073862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_181_layer_call_and_return_conditional_losses_25073613Ъ
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall*dense_181/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073720в
!dense_182/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_182_25073866dense_182_25073868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_182_layer_call_and_return_conditional_losses_25073636П
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_180_25073836*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: П
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_181_25073842*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_182_25073848*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_180_25073854*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181_25073860*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_182/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /ў
NoOpNoOp#^conv2d_180/StatefulPartitionedCall4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_181/StatefulPartitionedCall4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_182/StatefulPartitionedCall4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_180/StatefulPartitionedCall3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_181/StatefulPartitionedCall3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_182/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2H
"conv2d_180/StatefulPartitionedCall"conv2d_180/StatefulPartitionedCall2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_181/StatefulPartitionedCall"conv2d_181/StatefulPartitionedCall2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_182/StatefulPartitionedCall"conv2d_182/StatefulPartitionedCall2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_180/StatefulPartitionedCall!dense_180/StatefulPartitionedCall2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_181/StatefulPartitionedCall!dense_181/StatefulPartitionedCall2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_182/StatefulPartitionedCall!dense_182/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
й
╣
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25073556

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         АЫ
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Ан
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ЧI
╖
#__inference__wrapped_model_25073466
input_61M
3ocr_model_conv2d_180_conv2d_readvariableop_resource: B
4ocr_model_conv2d_180_biasadd_readvariableop_resource: M
3ocr_model_conv2d_181_conv2d_readvariableop_resource: @B
4ocr_model_conv2d_181_biasadd_readvariableop_resource:@N
3ocr_model_conv2d_182_conv2d_readvariableop_resource:@АC
4ocr_model_conv2d_182_biasadd_readvariableop_resource:	АE
2ocr_model_dense_180_matmul_readvariableop_resource:	А@A
3ocr_model_dense_180_biasadd_readvariableop_resource:@D
2ocr_model_dense_181_matmul_readvariableop_resource:@@A
3ocr_model_dense_181_biasadd_readvariableop_resource:@D
2ocr_model_dense_182_matmul_readvariableop_resource:@/A
3ocr_model_dense_182_biasadd_readvariableop_resource:/
identityИв+OCR_Model/conv2d_180/BiasAdd/ReadVariableOpв*OCR_Model/conv2d_180/Conv2D/ReadVariableOpв+OCR_Model/conv2d_181/BiasAdd/ReadVariableOpв*OCR_Model/conv2d_181/Conv2D/ReadVariableOpв+OCR_Model/conv2d_182/BiasAdd/ReadVariableOpв*OCR_Model/conv2d_182/Conv2D/ReadVariableOpв*OCR_Model/dense_180/BiasAdd/ReadVariableOpв)OCR_Model/dense_180/MatMul/ReadVariableOpв*OCR_Model/dense_181/BiasAdd/ReadVariableOpв)OCR_Model/dense_181/MatMul/ReadVariableOpв*OCR_Model/dense_182/BiasAdd/ReadVariableOpв)OCR_Model/dense_182/MatMul/ReadVariableOpж
*OCR_Model/conv2d_180/Conv2D/ReadVariableOpReadVariableOp3ocr_model_conv2d_180_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┼
OCR_Model/conv2d_180/Conv2DConv2Dinput_612OCR_Model/conv2d_180/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ь
+OCR_Model/conv2d_180/BiasAdd/ReadVariableOpReadVariableOp4ocr_model_conv2d_180_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╝
OCR_Model/conv2d_180/BiasAddBiasAdd$OCR_Model/conv2d_180/Conv2D:output:03OCR_Model/conv2d_180/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          В
OCR_Model/conv2d_180/ReluRelu%OCR_Model/conv2d_180/BiasAdd:output:0*
T0*/
_output_shapes
:          ─
#OCR_Model/max_pooling2d_120/MaxPoolMaxPool'OCR_Model/conv2d_180/Relu:activations:0*/
_output_shapes
:         		 *
ksize
*
paddingVALID*
strides
ж
*OCR_Model/conv2d_181/Conv2D/ReadVariableOpReadVariableOp3ocr_model_conv2d_181_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0щ
OCR_Model/conv2d_181/Conv2DConv2D,OCR_Model/max_pooling2d_120/MaxPool:output:02OCR_Model/conv2d_181/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingSAME*
strides
Ь
+OCR_Model/conv2d_181/BiasAdd/ReadVariableOpReadVariableOp4ocr_model_conv2d_181_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╝
OCR_Model/conv2d_181/BiasAddBiasAdd$OCR_Model/conv2d_181/Conv2D:output:03OCR_Model/conv2d_181/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@В
OCR_Model/conv2d_181/ReluRelu%OCR_Model/conv2d_181/BiasAdd:output:0*
T0*/
_output_shapes
:         		@─
#OCR_Model/max_pooling2d_121/MaxPoolMaxPool'OCR_Model/conv2d_181/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
з
*OCR_Model/conv2d_182/Conv2D/ReadVariableOpReadVariableOp3ocr_model_conv2d_182_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0ы
OCR_Model/conv2d_182/Conv2DConv2D,OCR_Model/max_pooling2d_121/MaxPool:output:02OCR_Model/conv2d_182/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
Э
+OCR_Model/conv2d_182/BiasAdd/ReadVariableOpReadVariableOp4ocr_model_conv2d_182_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╜
OCR_Model/conv2d_182/BiasAddBiasAdd$OCR_Model/conv2d_182/Conv2D:output:03OCR_Model/conv2d_182/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АГ
OCR_Model/conv2d_182/ReluRelu%OCR_Model/conv2d_182/BiasAdd:output:0*
T0*0
_output_shapes
:         Аk
OCR_Model/flatten_60/ConstConst*
_output_shapes
:*
dtype0*
valueB"       и
OCR_Model/flatten_60/ReshapeReshape'OCR_Model/conv2d_182/Relu:activations:0#OCR_Model/flatten_60/Const:output:0*
T0*(
_output_shapes
:         АЭ
)OCR_Model/dense_180/MatMul/ReadVariableOpReadVariableOp2ocr_model_dense_180_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0░
OCR_Model/dense_180/MatMulMatMul%OCR_Model/flatten_60/Reshape:output:01OCR_Model/dense_180/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
*OCR_Model/dense_180/BiasAdd/ReadVariableOpReadVariableOp3ocr_model_dense_180_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▓
OCR_Model/dense_180/BiasAddBiasAdd$OCR_Model/dense_180/MatMul:product:02OCR_Model/dense_180/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
OCR_Model/dense_180/ReluRelu$OCR_Model/dense_180/BiasAdd:output:0*
T0*'
_output_shapes
:         @Г
OCR_Model/dropout_37/IdentityIdentity&OCR_Model/dense_180/Relu:activations:0*
T0*'
_output_shapes
:         @Ь
)OCR_Model/dense_181/MatMul/ReadVariableOpReadVariableOp2ocr_model_dense_181_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0▒
OCR_Model/dense_181/MatMulMatMul&OCR_Model/dropout_37/Identity:output:01OCR_Model/dense_181/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
*OCR_Model/dense_181/BiasAdd/ReadVariableOpReadVariableOp3ocr_model_dense_181_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▓
OCR_Model/dense_181/BiasAddBiasAdd$OCR_Model/dense_181/MatMul:product:02OCR_Model/dense_181/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
OCR_Model/dense_181/ReluRelu$OCR_Model/dense_181/BiasAdd:output:0*
T0*'
_output_shapes
:         @Г
OCR_Model/dropout_38/IdentityIdentity&OCR_Model/dense_181/Relu:activations:0*
T0*'
_output_shapes
:         @Ь
)OCR_Model/dense_182/MatMul/ReadVariableOpReadVariableOp2ocr_model_dense_182_matmul_readvariableop_resource*
_output_shapes

:@/*
dtype0▒
OCR_Model/dense_182/MatMulMatMul&OCR_Model/dropout_38/Identity:output:01OCR_Model/dense_182/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /Ъ
*OCR_Model/dense_182/BiasAdd/ReadVariableOpReadVariableOp3ocr_model_dense_182_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0▓
OCR_Model/dense_182/BiasAddBiasAdd$OCR_Model/dense_182/MatMul:product:02OCR_Model/dense_182/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /s
IdentityIdentity$OCR_Model/dense_182/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         /т
NoOpNoOp,^OCR_Model/conv2d_180/BiasAdd/ReadVariableOp+^OCR_Model/conv2d_180/Conv2D/ReadVariableOp,^OCR_Model/conv2d_181/BiasAdd/ReadVariableOp+^OCR_Model/conv2d_181/Conv2D/ReadVariableOp,^OCR_Model/conv2d_182/BiasAdd/ReadVariableOp+^OCR_Model/conv2d_182/Conv2D/ReadVariableOp+^OCR_Model/dense_180/BiasAdd/ReadVariableOp*^OCR_Model/dense_180/MatMul/ReadVariableOp+^OCR_Model/dense_181/BiasAdd/ReadVariableOp*^OCR_Model/dense_181/MatMul/ReadVariableOp+^OCR_Model/dense_182/BiasAdd/ReadVariableOp*^OCR_Model/dense_182/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2Z
+OCR_Model/conv2d_180/BiasAdd/ReadVariableOp+OCR_Model/conv2d_180/BiasAdd/ReadVariableOp2X
*OCR_Model/conv2d_180/Conv2D/ReadVariableOp*OCR_Model/conv2d_180/Conv2D/ReadVariableOp2Z
+OCR_Model/conv2d_181/BiasAdd/ReadVariableOp+OCR_Model/conv2d_181/BiasAdd/ReadVariableOp2X
*OCR_Model/conv2d_181/Conv2D/ReadVariableOp*OCR_Model/conv2d_181/Conv2D/ReadVariableOp2Z
+OCR_Model/conv2d_182/BiasAdd/ReadVariableOp+OCR_Model/conv2d_182/BiasAdd/ReadVariableOp2X
*OCR_Model/conv2d_182/Conv2D/ReadVariableOp*OCR_Model/conv2d_182/Conv2D/ReadVariableOp2X
*OCR_Model/dense_180/BiasAdd/ReadVariableOp*OCR_Model/dense_180/BiasAdd/ReadVariableOp2V
)OCR_Model/dense_180/MatMul/ReadVariableOp)OCR_Model/dense_180/MatMul/ReadVariableOp2X
*OCR_Model/dense_181/BiasAdd/ReadVariableOp*OCR_Model/dense_181/BiasAdd/ReadVariableOp2V
)OCR_Model/dense_181/MatMul/ReadVariableOp)OCR_Model/dense_181/MatMul/ReadVariableOp2X
*OCR_Model/dense_182/BiasAdd/ReadVariableOp*OCR_Model/dense_182/BiasAdd/ReadVariableOp2V
)OCR_Model/dense_182/MatMul/ReadVariableOp)OCR_Model/dense_182/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         
"
_user_specified_name
input_61
В

g
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074562

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╝
I
-__inference_flatten_60_layer_call_fn_25074454

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_60_layer_call_and_return_conditional_losses_25073568a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
∙
д
-__inference_conv2d_182_layer_call_fn_25074434

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25073556x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
й
╣
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25074449

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         АЫ
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Ан
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
─┤
Г
$__inference__traced_restore_25074929
file_prefix<
"assignvariableop_conv2d_180_kernel: 0
"assignvariableop_1_conv2d_180_bias: >
$assignvariableop_2_conv2d_181_kernel: @0
"assignvariableop_3_conv2d_181_bias:@?
$assignvariableop_4_conv2d_182_kernel:@А1
"assignvariableop_5_conv2d_182_bias:	А6
#assignvariableop_6_dense_180_kernel:	А@/
!assignvariableop_7_dense_180_bias:@5
#assignvariableop_8_dense_181_kernel:@@/
!assignvariableop_9_dense_181_bias:@6
$assignvariableop_10_dense_182_kernel:@/0
"assignvariableop_11_dense_182_bias:/'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: F
,assignvariableop_21_adam_conv2d_180_kernel_m: 8
*assignvariableop_22_adam_conv2d_180_bias_m: F
,assignvariableop_23_adam_conv2d_181_kernel_m: @8
*assignvariableop_24_adam_conv2d_181_bias_m:@G
,assignvariableop_25_adam_conv2d_182_kernel_m:@А9
*assignvariableop_26_adam_conv2d_182_bias_m:	А>
+assignvariableop_27_adam_dense_180_kernel_m:	А@7
)assignvariableop_28_adam_dense_180_bias_m:@=
+assignvariableop_29_adam_dense_181_kernel_m:@@7
)assignvariableop_30_adam_dense_181_bias_m:@=
+assignvariableop_31_adam_dense_182_kernel_m:@/7
)assignvariableop_32_adam_dense_182_bias_m:/F
,assignvariableop_33_adam_conv2d_180_kernel_v: 8
*assignvariableop_34_adam_conv2d_180_bias_v: F
,assignvariableop_35_adam_conv2d_181_kernel_v: @8
*assignvariableop_36_adam_conv2d_181_bias_v:@G
,assignvariableop_37_adam_conv2d_182_kernel_v:@А9
*assignvariableop_38_adam_conv2d_182_bias_v:	А>
+assignvariableop_39_adam_dense_180_kernel_v:	А@7
)assignvariableop_40_adam_dense_180_bias_v:@=
+assignvariableop_41_adam_dense_181_kernel_v:@@7
)assignvariableop_42_adam_dense_181_bias_v:@=
+assignvariableop_43_adam_dense_182_kernel_v:@/7
)assignvariableop_44_adam_dense_182_bias_v:/
identity_46ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╠
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B З
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
╕::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_180_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_180_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_181_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_181_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_182_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_182_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_180_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_180_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_181_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_181_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_182_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_182_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_conv2d_180_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_conv2d_180_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_conv2d_181_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv2d_181_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_conv2d_182_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_conv2d_182_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_180_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_180_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_181_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_181_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_182_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_182_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_180_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_180_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_181_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_181_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_conv2d_182_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_conv2d_182_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_180_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_180_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_181_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_181_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_dense_182_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_182_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 н
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: Ъ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
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
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
█
f
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073596

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я
╖
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25073534

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         		@Ъ
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         		@н
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         		 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         		 
 
_user_specified_nameinputs
Ч
k
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25073487

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
в
┼
,__inference_OCR_Model_layer_call_fn_25073690
input_61!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@/

unknown_10:/
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25073663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_61
щQ
ъ
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074066
input_61-
conv2d_180_25074010: !
conv2d_180_25074012: -
conv2d_181_25074016: @!
conv2d_181_25074018:@.
conv2d_182_25074022:@А"
conv2d_182_25074024:	А%
dense_180_25074028:	А@ 
dense_180_25074030:@$
dense_181_25074034:@@ 
dense_181_25074036:@$
dense_182_25074040:@/ 
dense_182_25074042:/
identityИв"conv2d_180/StatefulPartitionedCallв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_181/StatefulPartitionedCallв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_182/StatefulPartitionedCallв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_180/StatefulPartitionedCallв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_181/StatefulPartitionedCallв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_182/StatefulPartitionedCallв"dropout_37/StatefulPartitionedCallв"dropout_38/StatefulPartitionedCallЛ
"conv2d_180/StatefulPartitionedCallStatefulPartitionedCallinput_61conv2d_180_25074010conv2d_180_25074012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25073512№
!max_pooling2d_120/PartitionedCallPartitionedCall+conv2d_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25073475н
"conv2d_181/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_120/PartitionedCall:output:0conv2d_181_25074016conv2d_181_25074018*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25073534№
!max_pooling2d_121/PartitionedCallPartitionedCall+conv2d_181/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25073487о
"conv2d_182/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_121/PartitionedCall:output:0conv2d_182_25074022conv2d_182_25074024*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25073556ч
flatten_60/PartitionedCallPartitionedCall+conv2d_182/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_60_layer_call_and_return_conditional_losses_25073568Ъ
!dense_180/StatefulPartitionedCallStatefulPartitionedCall#flatten_60/PartitionedCall:output:0dense_180_25074028dense_180_25074030*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_180_layer_call_and_return_conditional_losses_25073585ї
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall*dense_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073753в
!dense_181/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_181_25074034dense_181_25074036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_181_layer_call_and_return_conditional_losses_25073613Ъ
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall*dense_181/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073720в
!dense_182/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_182_25074040dense_182_25074042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_182_layer_call_and_return_conditional_losses_25073636П
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_180_25074010*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: П
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_181_25074016*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_182_25074022*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_180_25074028*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181_25074034*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_182/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /ў
NoOpNoOp#^conv2d_180/StatefulPartitionedCall4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_181/StatefulPartitionedCall4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_182/StatefulPartitionedCall4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_180/StatefulPartitionedCall3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_181/StatefulPartitionedCall3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_182/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2H
"conv2d_180/StatefulPartitionedCall"conv2d_180/StatefulPartitionedCall2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_181/StatefulPartitionedCall"conv2d_181/StatefulPartitionedCall2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_182/StatefulPartitionedCall"conv2d_182/StatefulPartitionedCall2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_180/StatefulPartitionedCall!dense_180/StatefulPartitionedCall2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_181/StatefulPartitionedCall!dense_181/StatefulPartitionedCall2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_182/StatefulPartitionedCall!dense_182/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_61
г
н
G__inference_dense_181_layer_call_and_return_conditional_losses_25073613

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @С
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @м
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
и
I
-__inference_dropout_38_layer_call_fn_25074540

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073624`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
В

g
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073753

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
В

g
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074511

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я
╖
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25073512

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          Ъ
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          н
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
цN
Ю
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25073663

inputs-
conv2d_180_25073513: !
conv2d_180_25073515: -
conv2d_181_25073535: @!
conv2d_181_25073537:@.
conv2d_182_25073557:@А"
conv2d_182_25073559:	А%
dense_180_25073586:	А@ 
dense_180_25073588:@$
dense_181_25073614:@@ 
dense_181_25073616:@$
dense_182_25073637:@/ 
dense_182_25073639:/
identityИв"conv2d_180/StatefulPartitionedCallв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_181/StatefulPartitionedCallв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_182/StatefulPartitionedCallв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_180/StatefulPartitionedCallв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_181/StatefulPartitionedCallв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_182/StatefulPartitionedCallЙ
"conv2d_180/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_180_25073513conv2d_180_25073515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25073512№
!max_pooling2d_120/PartitionedCallPartitionedCall+conv2d_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25073475н
"conv2d_181/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_120/PartitionedCall:output:0conv2d_181_25073535conv2d_181_25073537*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25073534№
!max_pooling2d_121/PartitionedCallPartitionedCall+conv2d_181/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25073487о
"conv2d_182/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_121/PartitionedCall:output:0conv2d_182_25073557conv2d_182_25073559*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25073556ч
flatten_60/PartitionedCallPartitionedCall+conv2d_182/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_60_layer_call_and_return_conditional_losses_25073568Ъ
!dense_180/StatefulPartitionedCallStatefulPartitionedCall#flatten_60/PartitionedCall:output:0dense_180_25073586dense_180_25073588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_180_layer_call_and_return_conditional_losses_25073585х
dropout_37/PartitionedCallPartitionedCall*dense_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073596Ъ
!dense_181/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_181_25073614dense_181_25073616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_181_layer_call_and_return_conditional_losses_25073613х
dropout_38/PartitionedCallPartitionedCall*dense_181/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073624Ъ
!dense_182/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_182_25073637dense_182_25073639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_182_layer_call_and_return_conditional_losses_25073636П
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_180_25073513*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: П
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_181_25073535*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_182_25073557*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_180_25073586*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181_25073614*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_182/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /н
NoOpNoOp#^conv2d_180/StatefulPartitionedCall4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_181/StatefulPartitionedCall4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_182/StatefulPartitionedCall4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_180/StatefulPartitionedCall3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_181/StatefulPartitionedCall3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_182/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2H
"conv2d_180/StatefulPartitionedCall"conv2d_180/StatefulPartitionedCall2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_181/StatefulPartitionedCall"conv2d_181/StatefulPartitionedCall2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_182/StatefulPartitionedCall"conv2d_182/StatefulPartitionedCall2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_180/StatefulPartitionedCall!dense_180/StatefulPartitionedCall2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_181/StatefulPartitionedCall!dense_181/StatefulPartitionedCall2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_182/StatefulPartitionedCall!dense_182/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ї
в
-__inference_conv2d_180_layer_call_fn_25074366

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25073512w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╦
Щ
,__inference_dense_182_layer_call_fn_25074571

inputs
unknown:@/
	unknown_0:/
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_182_layer_call_and_return_conditional_losses_25073636o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
т\
ф
!__inference__traced_save_25074784
file_prefix0
,savev2_conv2d_180_kernel_read_readvariableop.
*savev2_conv2d_180_bias_read_readvariableop0
,savev2_conv2d_181_kernel_read_readvariableop.
*savev2_conv2d_181_bias_read_readvariableop0
,savev2_conv2d_182_kernel_read_readvariableop.
*savev2_conv2d_182_bias_read_readvariableop/
+savev2_dense_180_kernel_read_readvariableop-
)savev2_dense_180_bias_read_readvariableop/
+savev2_dense_181_kernel_read_readvariableop-
)savev2_dense_181_bias_read_readvariableop/
+savev2_dense_182_kernel_read_readvariableop-
)savev2_dense_182_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_180_kernel_m_read_readvariableop5
1savev2_adam_conv2d_180_bias_m_read_readvariableop7
3savev2_adam_conv2d_181_kernel_m_read_readvariableop5
1savev2_adam_conv2d_181_bias_m_read_readvariableop7
3savev2_adam_conv2d_182_kernel_m_read_readvariableop5
1savev2_adam_conv2d_182_bias_m_read_readvariableop6
2savev2_adam_dense_180_kernel_m_read_readvariableop4
0savev2_adam_dense_180_bias_m_read_readvariableop6
2savev2_adam_dense_181_kernel_m_read_readvariableop4
0savev2_adam_dense_181_bias_m_read_readvariableop6
2savev2_adam_dense_182_kernel_m_read_readvariableop4
0savev2_adam_dense_182_bias_m_read_readvariableop7
3savev2_adam_conv2d_180_kernel_v_read_readvariableop5
1savev2_adam_conv2d_180_bias_v_read_readvariableop7
3savev2_adam_conv2d_181_kernel_v_read_readvariableop5
1savev2_adam_conv2d_181_bias_v_read_readvariableop7
3savev2_adam_conv2d_182_kernel_v_read_readvariableop5
1savev2_adam_conv2d_182_bias_v_read_readvariableop6
2savev2_adam_dense_180_kernel_v_read_readvariableop4
0savev2_adam_dense_180_bias_v_read_readvariableop6
2savev2_adam_dense_181_kernel_v_read_readvariableop4
0savev2_adam_dense_181_bias_v_read_readvariableop6
2savev2_adam_dense_182_kernel_v_read_readvariableop4
0savev2_adam_dense_182_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: г
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╔
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Я
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_180_kernel_read_readvariableop*savev2_conv2d_180_bias_read_readvariableop,savev2_conv2d_181_kernel_read_readvariableop*savev2_conv2d_181_bias_read_readvariableop,savev2_conv2d_182_kernel_read_readvariableop*savev2_conv2d_182_bias_read_readvariableop+savev2_dense_180_kernel_read_readvariableop)savev2_dense_180_bias_read_readvariableop+savev2_dense_181_kernel_read_readvariableop)savev2_dense_181_bias_read_readvariableop+savev2_dense_182_kernel_read_readvariableop)savev2_dense_182_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_180_kernel_m_read_readvariableop1savev2_adam_conv2d_180_bias_m_read_readvariableop3savev2_adam_conv2d_181_kernel_m_read_readvariableop1savev2_adam_conv2d_181_bias_m_read_readvariableop3savev2_adam_conv2d_182_kernel_m_read_readvariableop1savev2_adam_conv2d_182_bias_m_read_readvariableop2savev2_adam_dense_180_kernel_m_read_readvariableop0savev2_adam_dense_180_bias_m_read_readvariableop2savev2_adam_dense_181_kernel_m_read_readvariableop0savev2_adam_dense_181_bias_m_read_readvariableop2savev2_adam_dense_182_kernel_m_read_readvariableop0savev2_adam_dense_182_bias_m_read_readvariableop3savev2_adam_conv2d_180_kernel_v_read_readvariableop1savev2_adam_conv2d_180_bias_v_read_readvariableop3savev2_adam_conv2d_181_kernel_v_read_readvariableop1savev2_adam_conv2d_181_bias_v_read_readvariableop3savev2_adam_conv2d_182_kernel_v_read_readvariableop1savev2_adam_conv2d_182_bias_v_read_readvariableop2savev2_adam_dense_180_kernel_v_read_readvariableop0savev2_adam_dense_180_bias_v_read_readvariableop2savev2_adam_dense_181_kernel_v_read_readvariableop0savev2_adam_dense_181_bias_v_read_readvariableop2savev2_adam_dense_182_kernel_v_read_readvariableop0savev2_adam_dense_182_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ь
_input_shapesК
З: : : : @:@:@А:А:	А@:@:@@:@:@/:/: : : : : : : : : : : : @:@:@А:А:	А@:@:@@:@:@/:/: : : @:@:@А:А:	А@:@:@@:@:@/:/: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@/: 

_output_shapes
:/:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$  

_output_shapes

:@/: !

_output_shapes
:/:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@А:!'

_output_shapes	
:А:%(!

_output_shapes
:	А@: )

_output_shapes
:@:$* 

_output_shapes

:@@: +

_output_shapes
:@:$, 

_output_shapes

:@/: -

_output_shapes
:/:.

_output_shapes
: 
█
f
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073624

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ь
├
,__inference_OCR_Model_layer_call_fn_25074201

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@/

unknown_10:/
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25073892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Я
╖
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25074415

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         		@Ъ
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         		@н
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         		 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         		 
 
_user_specified_nameinputs
Ь
├
,__inference_OCR_Model_layer_call_fn_25074172

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@/

unknown_10:/
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25073663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
P
4__inference_max_pooling2d_121_layer_call_fn_25074420

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25073487Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ч
k
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25074425

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
·
f
-__inference_dropout_37_layer_call_fn_25074494

inputs
identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073753o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
и
о
G__inference_dense_180_layer_call_and_return_conditional_losses_25073585

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Т
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @м
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
и
I
-__inference_dropout_37_layer_call_fn_25074489

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073596`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ьN
а
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074007
input_61-
conv2d_180_25073951: !
conv2d_180_25073953: -
conv2d_181_25073957: @!
conv2d_181_25073959:@.
conv2d_182_25073963:@А"
conv2d_182_25073965:	А%
dense_180_25073969:	А@ 
dense_180_25073971:@$
dense_181_25073975:@@ 
dense_181_25073977:@$
dense_182_25073981:@/ 
dense_182_25073983:/
identityИв"conv2d_180/StatefulPartitionedCallв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_181/StatefulPartitionedCallв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpв"conv2d_182/StatefulPartitionedCallв3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_180/StatefulPartitionedCallв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_181/StatefulPartitionedCallв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpв!dense_182/StatefulPartitionedCallЛ
"conv2d_180/StatefulPartitionedCallStatefulPartitionedCallinput_61conv2d_180_25073951conv2d_180_25073953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25073512№
!max_pooling2d_120/PartitionedCallPartitionedCall+conv2d_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25073475н
"conv2d_181/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_120/PartitionedCall:output:0conv2d_181_25073957conv2d_181_25073959*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25073534№
!max_pooling2d_121/PartitionedCallPartitionedCall+conv2d_181/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25073487о
"conv2d_182/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_121/PartitionedCall:output:0conv2d_182_25073963conv2d_182_25073965*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25073556ч
flatten_60/PartitionedCallPartitionedCall+conv2d_182/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_60_layer_call_and_return_conditional_losses_25073568Ъ
!dense_180/StatefulPartitionedCallStatefulPartitionedCall#flatten_60/PartitionedCall:output:0dense_180_25073969dense_180_25073971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_180_layer_call_and_return_conditional_losses_25073585х
dropout_37/PartitionedCallPartitionedCall*dense_180/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_37_layer_call_and_return_conditional_losses_25073596Ъ
!dense_181/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_181_25073975dense_181_25073977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_181_layer_call_and_return_conditional_losses_25073613х
dropout_38/PartitionedCallPartitionedCall*dense_181/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073624Ъ
!dense_182/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_182_25073981dense_182_25073983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_182_layer_call_and_return_conditional_losses_25073636П
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_180_25073951*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: П
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_181_25073957*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_182_25073963*'
_output_shapes
:@А*
dtype0М
$conv2d_182/kernel/Regularizer/L2LossL2Loss;conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_182/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_182/kernel/Regularizer/mulMul,conv2d_182/kernel/Regularizer/mul/x:output:0-conv2d_182/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_180_25073969*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181_25073975*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_182/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /н
NoOpNoOp#^conv2d_180/StatefulPartitionedCall4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_181/StatefulPartitionedCall4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp#^conv2d_182/StatefulPartitionedCall4^conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_180/StatefulPartitionedCall3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_181/StatefulPartitionedCall3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_182/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 2H
"conv2d_180/StatefulPartitionedCall"conv2d_180/StatefulPartitionedCall2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_181/StatefulPartitionedCall"conv2d_181/StatefulPartitionedCall2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp2H
"conv2d_182/StatefulPartitionedCall"conv2d_182/StatefulPartitionedCall2j
3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_182/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_180/StatefulPartitionedCall!dense_180/StatefulPartitionedCall2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_181/StatefulPartitionedCall!dense_181/StatefulPartitionedCall2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_182/StatefulPartitionedCall!dense_182/StatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_61
╬	
┐
__inference_loss_fn_0_25074590V
<conv2d_180_kernel_regularizer_l2loss_readvariableop_resource: 
identityИв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp╕
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<conv2d_180_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_180/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp
Ч
k
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25074391

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
В

g
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073720

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╠
d
H__inference_flatten_60_layer_call_and_return_conditional_losses_25073568

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╦
Щ
,__inference_dense_181_layer_call_fn_25074520

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_181_layer_call_and_return_conditional_losses_25073613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
в
┼
,__inference_OCR_Model_layer_call_fn_25073948
input_61!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@/

unknown_10:/
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         /*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25073892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         /`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_61
╬	
┐
__inference_loss_fn_1_25074599V
<conv2d_181_kernel_regularizer_l2loss_readvariableop_resource: @
identityИв3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp╕
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<conv2d_181_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: @*
dtype0М
$conv2d_181/kernel/Regularizer/L2LossL2Loss;conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_181/kernel/Regularizer/mulMul,conv2d_181/kernel/Regularizer/mul/x:output:0-conv2d_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_181/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_181/kernel/Regularizer/L2Loss/ReadVariableOp
Я
╖
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25074381

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          Ъ
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0М
$conv2d_180/kernel/Regularizer/L2LossL2Loss;conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2ж
!conv2d_180/kernel/Regularizer/mulMul,conv2d_180/kernel/Regularizer/mul/x:output:0-conv2d_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          н
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp4^conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2j
3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp3conv2d_180/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ї
в
-__inference_conv2d_181_layer_call_fn_25074400

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25073534w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         		@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         		 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         		 
 
_user_specified_nameinputs
╬
Ъ
,__inference_dense_180_layer_call_fn_25074469

inputs
unknown:	А@
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_180_layer_call_and_return_conditional_losses_25073585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
f
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074550

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
░	
╡
__inference_loss_fn_4_25074626M
;dense_181_kernel_regularizer_l2loss_readvariableop_resource:@@
identityИв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpо
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_181_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_181/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp
г
н
G__inference_dense_181_layer_call_and_return_conditional_losses_25074535

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @С
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0К
#dense_181/kernel/Regularizer/L2LossL2Loss:dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_181/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_181/kernel/Regularizer/mulMul+dense_181/kernel/Regularizer/mul/x:output:0,dense_181/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @м
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_181/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp2dense_181/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
и
о
G__inference_dense_180_layer_call_and_return_conditional_losses_25074484

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Т
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @м
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╩	
°
G__inference_dense_182_layer_call_and_return_conditional_losses_25073636

inputs0
matmul_readvariableop_resource:@/-
biasadd_readvariableop_resource:/
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         /_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         /w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
·
f
-__inference_dropout_38_layer_call_fn_25074545

inputs
identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_38_layer_call_and_return_conditional_losses_25073720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
├
P
4__inference_max_pooling2d_120_layer_call_fn_25074386

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25073475Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ч
k
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25073475

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▓	
╢
__inference_loss_fn_3_25074617N
;dense_180_kernel_regularizer_l2loss_readvariableop_resource:	А@
identityИв2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpп
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_180_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	А@*
dtype0К
#dense_180/kernel/Regularizer/L2LossL2Loss:dense_180/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_180/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w╠л2г
 dense_180/kernel/Regularizer/mulMul+dense_180/kernel/Regularizer/mul/x:output:0,dense_180/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_180/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_180/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp2dense_180/kernel/Regularizer/L2Loss/ReadVariableOp"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╢
serving_defaultв
E
input_619
serving_default_input_61:0         =
	dense_1820
StatefulPartitionedCall:0         /tensorflow/serving/predict:┘з
Т
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
е
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op"
_tf_keras_layer
е
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op"
_tf_keras_layer
е
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
╝
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator"
_tf_keras_layer
╗
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias"
_tf_keras_layer
╝
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator"
_tf_keras_layer
╗
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias"
_tf_keras_layer
v
0
1
*2
+3
94
:5
H6
I7
W8
X9
f10
g11"
trackable_list_wrapper
v
0
1
*2
+3
94
:5
H6
I7
W8
X9
f10
g11"
trackable_list_wrapper
C
h0
i1
j2
k3
l4"
trackable_list_wrapper
╩
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
х
rtrace_0
strace_1
ttrace_2
utrace_32·
,__inference_OCR_Model_layer_call_fn_25073690
,__inference_OCR_Model_layer_call_fn_25074172
,__inference_OCR_Model_layer_call_fn_25074201
,__inference_OCR_Model_layer_call_fn_25073948┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zrtrace_0zstrace_1zttrace_2zutrace_3
╤
vtrace_0
wtrace_1
xtrace_2
ytrace_32ц
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074272
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074357
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074007
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074066┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zvtrace_0zwtrace_1zxtrace_2zytrace_3
╧B╠
#__inference__wrapped_model_25073466input_61"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
├
ziter

{beta_1

|beta_2
	}decay
~learning_ratemсmт*mу+mф9mх:mцHmчImшWmщXmъfmыgmьvэvю*vя+vЁ9vё:vЄHvєIvЇWvїXvЎfvўgv°"
	optimizer
,
serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
є
Еtrace_02╘
-__inference_conv2d_180_layer_call_fn_25074366в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
О
Жtrace_02я
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25074381в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
+:) 2conv2d_180/kernel
: 2conv2d_180/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
·
Мtrace_02█
4__inference_max_pooling2d_120_layer_call_fn_25074386в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
Х
Нtrace_02Ў
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25074391в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
▓
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
є
Уtrace_02╘
-__inference_conv2d_181_layer_call_fn_25074400в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
О
Фtrace_02я
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25074415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
+:) @2conv2d_181/kernel
:@2conv2d_181/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
·
Ъtrace_02█
4__inference_max_pooling2d_121_layer_call_fn_25074420в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
Х
Ыtrace_02Ў
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25074425в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
'
j0"
trackable_list_wrapper
▓
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
є
бtrace_02╘
-__inference_conv2d_182_layer_call_fn_25074434в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
О
вtrace_02я
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25074449в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
,:*@А2conv2d_182/kernel
:А2conv2d_182/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
є
иtrace_02╘
-__inference_flatten_60_layer_call_fn_25074454в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0
О
йtrace_02я
H__inference_flatten_60_layer_call_and_return_conditional_losses_25074460в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
'
k0"
trackable_list_wrapper
▓
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Є
пtrace_02╙
,__inference_dense_180_layer_call_fn_25074469в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0
Н
░trace_02ю
G__inference_dense_180_layer_call_and_return_conditional_losses_25074484в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z░trace_0
#:!	А@2dense_180/kernel
:@2dense_180/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
╧
╢trace_0
╖trace_12Ф
-__inference_dropout_37_layer_call_fn_25074489
-__inference_dropout_37_layer_call_fn_25074494│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╢trace_0z╖trace_1
Е
╕trace_0
╣trace_12╩
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074499
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074511│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0z╣trace_1
"
_generic_user_object
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
'
l0"
trackable_list_wrapper
▓
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Є
┐trace_02╙
,__inference_dense_181_layer_call_fn_25074520в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┐trace_0
Н
└trace_02ю
G__inference_dense_181_layer_call_and_return_conditional_losses_25074535в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z└trace_0
": @@2dense_181/kernel
:@2dense_181/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
╧
╞trace_0
╟trace_12Ф
-__inference_dropout_38_layer_call_fn_25074540
-__inference_dropout_38_layer_call_fn_25074545│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╞trace_0z╟trace_1
Е
╚trace_0
╔trace_12╩
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074550
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074562│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╚trace_0z╔trace_1
"
_generic_user_object
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
Є
╧trace_02╙
,__inference_dense_182_layer_call_fn_25074571в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╧trace_0
Н
╨trace_02ю
G__inference_dense_182_layer_call_and_return_conditional_losses_25074581в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0
": @/2dense_182/kernel
:/2dense_182/bias
╤
╤trace_02▓
__inference_loss_fn_0_25074590П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╤trace_0
╤
╥trace_02▓
__inference_loss_fn_1_25074599П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╥trace_0
╤
╙trace_02▓
__inference_loss_fn_2_25074608П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╙trace_0
╤
╘trace_02▓
__inference_loss_fn_3_25074617П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╘trace_0
╤
╒trace_02▓
__inference_loss_fn_4_25074626П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╒trace_0
 "
trackable_list_wrapper
n
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
10"
trackable_list_wrapper
0
╓0
╫1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 B№
,__inference_OCR_Model_layer_call_fn_25073690input_61"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_OCR_Model_layer_call_fn_25074172inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_OCR_Model_layer_call_fn_25074201inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
,__inference_OCR_Model_layer_call_fn_25073948input_61"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074272inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074357inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074007input_61"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074066input_61"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╬B╦
&__inference_signature_wrapper_25074123input_61"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
-__inference_conv2d_180_layer_call_fn_25074366inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25074381inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
шBх
4__inference_max_pooling2d_120_layer_call_fn_25074386inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ГBА
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25074391inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
-__inference_conv2d_181_layer_call_fn_25074400inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25074415inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
шBх
4__inference_max_pooling2d_121_layer_call_fn_25074420inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ГBА
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25074425inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
j0"
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
-__inference_conv2d_182_layer_call_fn_25074434inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25074449inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
сB▐
-__inference_flatten_60_layer_call_fn_25074454inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_flatten_60_layer_call_and_return_conditional_losses_25074460inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_dense_180_layer_call_fn_25074469inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_dense_180_layer_call_and_return_conditional_losses_25074484inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ЄBя
-__inference_dropout_37_layer_call_fn_25074489inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
-__inference_dropout_37_layer_call_fn_25074494inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074499inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074511inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
l0"
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_dense_181_layer_call_fn_25074520inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_dense_181_layer_call_and_return_conditional_losses_25074535inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ЄBя
-__inference_dropout_38_layer_call_fn_25074540inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
-__inference_dropout_38_layer_call_fn_25074545inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074550inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074562inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_dense_182_layer_call_fn_25074571inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_dense_182_layer_call_and_return_conditional_losses_25074581inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡B▓
__inference_loss_fn_0_25074590"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡B▓
__inference_loss_fn_1_25074599"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡B▓
__inference_loss_fn_2_25074608"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡B▓
__inference_loss_fn_3_25074617"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡B▓
__inference_loss_fn_4_25074626"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
R
╪	variables
┘	keras_api

┌total

█count"
_tf_keras_metric
c
▄	variables
▌	keras_api

▐total

▀count
р
_fn_kwargs"
_tf_keras_metric
0
┌0
█1"
trackable_list_wrapper
.
╪	variables"
_generic_user_object
:  (2total
:  (2count
0
▐0
▀1"
trackable_list_wrapper
.
▄	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0:. 2Adam/conv2d_180/kernel/m
":  2Adam/conv2d_180/bias/m
0:. @2Adam/conv2d_181/kernel/m
": @2Adam/conv2d_181/bias/m
1:/@А2Adam/conv2d_182/kernel/m
#:!А2Adam/conv2d_182/bias/m
(:&	А@2Adam/dense_180/kernel/m
!:@2Adam/dense_180/bias/m
':%@@2Adam/dense_181/kernel/m
!:@2Adam/dense_181/bias/m
':%@/2Adam/dense_182/kernel/m
!:/2Adam/dense_182/bias/m
0:. 2Adam/conv2d_180/kernel/v
":  2Adam/conv2d_180/bias/v
0:. @2Adam/conv2d_181/kernel/v
": @2Adam/conv2d_181/bias/v
1:/@А2Adam/conv2d_182/kernel/v
#:!А2Adam/conv2d_182/bias/v
(:&	А@2Adam/dense_180/kernel/v
!:@2Adam/dense_180/bias/v
':%@@2Adam/dense_181/kernel/v
!:@2Adam/dense_181/bias/v
':%@/2Adam/dense_182/kernel/v
!:/2Adam/dense_182/bias/v├
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074007x*+9:HIWXfgAв>
7в4
*К'
input_61         
p 

 
к "%в"
К
0         /
Ъ ├
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074066x*+9:HIWXfgAв>
7в4
*К'
input_61         
p

 
к "%в"
К
0         /
Ъ ┴
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074272v*+9:HIWXfg?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         /
Ъ ┴
G__inference_OCR_Model_layer_call_and_return_conditional_losses_25074357v*+9:HIWXfg?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         /
Ъ Ы
,__inference_OCR_Model_layer_call_fn_25073690k*+9:HIWXfgAв>
7в4
*К'
input_61         
p 

 
к "К         /Ы
,__inference_OCR_Model_layer_call_fn_25073948k*+9:HIWXfgAв>
7в4
*К'
input_61         
p

 
к "К         /Щ
,__inference_OCR_Model_layer_call_fn_25074172i*+9:HIWXfg?в<
5в2
(К%
inputs         
p 

 
к "К         /Щ
,__inference_OCR_Model_layer_call_fn_25074201i*+9:HIWXfg?в<
5в2
(К%
inputs         
p

 
к "К         /и
#__inference__wrapped_model_25073466А*+9:HIWXfg9в6
/в,
*К'
input_61         
к "5к2
0
	dense_182#К 
	dense_182         /╕
H__inference_conv2d_180_layer_call_and_return_conditional_losses_25074381l7в4
-в*
(К%
inputs         
к "-в*
#К 
0          
Ъ Р
-__inference_conv2d_180_layer_call_fn_25074366_7в4
-в*
(К%
inputs         
к " К          ╕
H__inference_conv2d_181_layer_call_and_return_conditional_losses_25074415l*+7в4
-в*
(К%
inputs         		 
к "-в*
#К 
0         		@
Ъ Р
-__inference_conv2d_181_layer_call_fn_25074400_*+7в4
-в*
(К%
inputs         		 
к " К         		@╣
H__inference_conv2d_182_layer_call_and_return_conditional_losses_25074449m9:7в4
-в*
(К%
inputs         @
к ".в+
$К!
0         А
Ъ С
-__inference_conv2d_182_layer_call_fn_25074434`9:7в4
-в*
(К%
inputs         @
к "!К         Аи
G__inference_dense_180_layer_call_and_return_conditional_losses_25074484]HI0в-
&в#
!К
inputs         А
к "%в"
К
0         @
Ъ А
,__inference_dense_180_layer_call_fn_25074469PHI0в-
&в#
!К
inputs         А
к "К         @з
G__inference_dense_181_layer_call_and_return_conditional_losses_25074535\WX/в,
%в"
 К
inputs         @
к "%в"
К
0         @
Ъ 
,__inference_dense_181_layer_call_fn_25074520OWX/в,
%в"
 К
inputs         @
к "К         @з
G__inference_dense_182_layer_call_and_return_conditional_losses_25074581\fg/в,
%в"
 К
inputs         @
к "%в"
К
0         /
Ъ 
,__inference_dense_182_layer_call_fn_25074571Ofg/в,
%в"
 К
inputs         @
к "К         /и
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074499\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ и
H__inference_dropout_37_layer_call_and_return_conditional_losses_25074511\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ А
-__inference_dropout_37_layer_call_fn_25074489O3в0
)в&
 К
inputs         @
p 
к "К         @А
-__inference_dropout_37_layer_call_fn_25074494O3в0
)в&
 К
inputs         @
p
к "К         @и
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074550\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ и
H__inference_dropout_38_layer_call_and_return_conditional_losses_25074562\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ А
-__inference_dropout_38_layer_call_fn_25074540O3в0
)в&
 К
inputs         @
p 
к "К         @А
-__inference_dropout_38_layer_call_fn_25074545O3в0
)в&
 К
inputs         @
p
к "К         @о
H__inference_flatten_60_layer_call_and_return_conditional_losses_25074460b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А
Ъ Ж
-__inference_flatten_60_layer_call_fn_25074454U8в5
.в+
)К&
inputs         А
к "К         А=
__inference_loss_fn_0_25074590в

в 
к "К =
__inference_loss_fn_1_25074599*в

в 
к "К =
__inference_loss_fn_2_250746089в

в 
к "К =
__inference_loss_fn_3_25074617Hв

в 
к "К =
__inference_loss_fn_4_25074626Wв

в 
к "К Є
O__inference_max_pooling2d_120_layer_call_and_return_conditional_losses_25074391ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╩
4__inference_max_pooling2d_120_layer_call_fn_25074386СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Є
O__inference_max_pooling2d_121_layer_call_and_return_conditional_losses_25074425ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╩
4__inference_max_pooling2d_121_layer_call_fn_25074420СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╖
&__inference_signature_wrapper_25074123М*+9:HIWXfgEвB
в 
;к8
6
input_61*К'
input_61         "5к2
0
	dense_182#К 
	dense_182         /