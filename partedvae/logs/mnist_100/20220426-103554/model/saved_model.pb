ľ<
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
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
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	
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
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??6
v
u_prior_meansVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_nameu_prior_means
o
!u_prior_means/Read/ReadVariableOpReadVariableOpu_prior_means*
_output_shapes

:
*
dtype0
?
u_prior_logvars_before_tanhVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*,
shared_nameu_prior_logvars_before_tanh
?
/u_prior_logvars_before_tanh/Read/ReadVariableOpReadVariableOpu_prior_logvars_before_tanh*
_output_shapes

:
*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?
*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:
*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	
?*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
??*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:  *
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_2/kernel
?
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?n
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?m
value?mB?m B?m
?
feature_extractor
z_mean_layer
z_logvar_layer
h_to_c_logit
c_to_a_logit
u_mean_layer
u_logvar_layer
decoder
	u_prior_means

u_prior_logvars_before_tanh
regularization_losses
trainable_variables
	variables
	keras_api

signatures
j
	nChannels
net
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
j
:	nChannels
;net
<regularization_losses
=trainable_variables
>	variables
?	keras_api
KI
VARIABLE_VALUEu_prior_means(u_prior_means/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEu_prior_logvars_before_tanh6u_prior_logvars_before_tanh/.ATTRIBUTES/VARIABLE_VALUE
 
?
@0
A1
B2
C3
D4
E5
F6
G7
8
9
10
11
"12
#13
(14
)15
.16
/17
418
519
H20
I21
J22
K23
L24
M25
N26
O27
P28
Q29
	30

31
?
@0
A1
B2
C3
D4
E5
F6
G7
8
9
10
11
"12
#13
(14
)15
.16
/17
418
519
H20
I21
J22
K23
L24
M25
N26
O27
P28
Q29
	30

31
?
regularization_losses
trainable_variables
Rmetrics
Slayer_metrics
Tnon_trainable_variables
	variables
Ulayer_regularization_losses

Vlayers
 
 
?
Wlayer_with_weights-0
Wlayer-0
Xlayer-1
Ylayer_with_weights-1
Ylayer-2
Zlayer-3
[layer_with_weights-2
[layer-4
\layer-5
]layer-6
^layer_with_weights-3
^layer-7
_layer-8
`regularization_losses
atrainable_variables
b	variables
c	keras_api
 
8
@0
A1
B2
C3
D4
E5
F6
G7
8
@0
A1
B2
C3
D4
E5
F6
G7
?
regularization_losses
trainable_variables
dmetrics
elayer_metrics
fnon_trainable_variables
	variables
glayer_regularization_losses

hlayers
RP
VARIABLE_VALUEdense_1/kernel.z_mean_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdense_1/bias,z_mean_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
trainable_variables
imetrics
jlayer_metrics
knon_trainable_variables
	variables
llayer_regularization_losses

mlayers
TR
VARIABLE_VALUEdense_2/kernel0z_logvar_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdense_2/bias.z_logvar_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
trainable_variables
nmetrics
olayer_metrics
pnon_trainable_variables
 	variables
qlayer_regularization_losses

rlayers
RP
VARIABLE_VALUEdense_3/kernel.h_to_c_logit/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdense_3/bias,h_to_c_logit/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
$regularization_losses
%trainable_variables
smetrics
tlayer_metrics
unon_trainable_variables
&	variables
vlayer_regularization_losses

wlayers
RP
VARIABLE_VALUEdense_4/kernel.c_to_a_logit/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdense_4/bias,c_to_a_logit/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
*regularization_losses
+trainable_variables
xmetrics
ylayer_metrics
znon_trainable_variables
,	variables
{layer_regularization_losses

|layers
RP
VARIABLE_VALUEdense_5/kernel.u_mean_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdense_5/bias,u_mean_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
0regularization_losses
1trainable_variables
}metrics
~layer_metrics
non_trainable_variables
2	variables
 ?layer_regularization_losses
?layers
TR
VARIABLE_VALUEdense_6/kernel0u_logvar_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdense_6/bias.u_logvar_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
6regularization_losses
7trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
8	variables
 ?layer_regularization_losses
?layers
 
?
?layer_with_weights-0
?layer-0
?layer-1
?layer_with_weights-1
?layer-2
?layer-3
?layer-4
?layer_with_weights-2
?layer-5
?layer-6
?layer_with_weights-3
?layer-7
?layer-8
?layer_with_weights-4
?layer-9
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
F
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
F
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
?
<regularization_losses
=trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
>	variables
 ?layer_regularization_losses
?layers
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_7/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_7/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_8/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_8/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv2d_transpose/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_1/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_1/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_2/kernel1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_2/bias1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
8
0
1
2
3
4
5
6
7
l

@kernel
Abias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Bkernel
Cbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Dkernel
Ebias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Fkernel
Gbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
8
@0
A1
B2
C3
D4
E5
F6
G7
8
@0
A1
B2
C3
D4
E5
F6
G7
?
`regularization_losses
atrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
b	variables
 ?layer_regularization_losses
?layers
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l

Hkernel
Ibias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Jkernel
Kbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Lkernel
Mbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Nkernel
Obias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Pkernel
Qbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
F
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
F
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

;0
 

@0
A1

@0
A1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

B0
C1

B0
C1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

D0
E1

D0
E1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

F0
G1

F0
G1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
 

H0
I1

H0
I1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

J0
K1

J0
K1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

L0
M1

L0
M1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

N0
O1

N0
O1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 

P0
Q1

P0
Q1
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
P
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:?????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? */
f*R(
&__inference_signature_wrapper_86805178
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!u_prior_means/Read/ReadVariableOp/u_prior_logvars_before_tanh/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOpConst*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? **
f%R#
!__inference__traced_save_86809402
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameu_prior_meansu_prior_logvars_before_tanhdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *-
f(R&
$__inference__traced_restore_86809508??4
??
?	
__inference_call_86808540
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeShape'sequential_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape'sequential_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape'sequential_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_namex
??
?
$__inference__traced_restore_86809508
file_prefix"
assignvariableop_u_prior_means2
.assignvariableop_1_u_prior_logvars_before_tanh%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias&
"assignvariableop_10_dense_5_kernel$
 assignvariableop_11_dense_5_bias&
"assignvariableop_12_dense_6_kernel$
 assignvariableop_13_dense_6_bias%
!assignvariableop_14_conv2d_kernel#
assignvariableop_15_conv2d_bias'
#assignvariableop_16_conv2d_1_kernel%
!assignvariableop_17_conv2d_1_bias'
#assignvariableop_18_conv2d_2_kernel%
!assignvariableop_19_conv2d_2_bias$
 assignvariableop_20_dense_kernel"
assignvariableop_21_dense_bias&
"assignvariableop_22_dense_7_kernel$
 assignvariableop_23_dense_7_bias&
"assignvariableop_24_dense_8_kernel$
 assignvariableop_25_dense_8_bias/
+assignvariableop_26_conv2d_transpose_kernel-
)assignvariableop_27_conv2d_transpose_bias1
-assignvariableop_28_conv2d_transpose_1_kernel/
+assignvariableop_29_conv2d_transpose_1_bias1
-assignvariableop_30_conv2d_transpose_2_kernel/
+assignvariableop_31_conv2d_transpose_2_bias
identity_33??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B(u_prior_means/.ATTRIBUTES/VARIABLE_VALUEB6u_prior_logvars_before_tanh/.ATTRIBUTES/VARIABLE_VALUEB.z_mean_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,z_mean_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0z_logvar_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB.z_logvar_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB.h_to_c_logit/kernel/.ATTRIBUTES/VARIABLE_VALUEB,h_to_c_logit/bias/.ATTRIBUTES/VARIABLE_VALUEB.c_to_a_logit/kernel/.ATTRIBUTES/VARIABLE_VALUEB,c_to_a_logit/bias/.ATTRIBUTES/VARIABLE_VALUEB.u_mean_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,u_mean_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0u_logvar_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB.u_logvar_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_u_prior_meansIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp.assignvariableop_1_u_prior_logvars_before_tanhIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conv2d_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_7_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_7_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_8_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_8_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_conv2d_transpose_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_conv2d_transpose_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp-assignvariableop_28_conv2d_transpose_1_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_conv2d_transpose_1_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_conv2d_transpose_2_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_conv2d_transpose_2_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32?
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
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
?.
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803895

inputs
dense_7_86803864
dense_7_86803866
dense_8_86803870
dense_8_86803872
conv2d_transpose_86803877
conv2d_transpose_86803879
conv2d_transpose_1_86803883
conv2d_transpose_1_86803885
conv2d_transpose_2_86803889
conv2d_transpose_2_86803891
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_86803864dense_7_86803866*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_868036922!
dense_7/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_868037132
re_lu_2/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_8_86803870dense_8_86803872*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_868037312!
dense_8/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_868037522
re_lu_3/PartitionedCall?
reshape/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_868037742
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_86803877conv2d_transpose_86803879*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_868035792*
(conv2d_transpose/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_868037922
re_lu_4/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_transpose_1_86803883conv2d_transpose_1_86803885*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_868036232,
*conv2d_transpose_1/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_868038102
re_lu_5/PartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_transpose_2_86803889conv2d_transpose_2_86803891*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_868036682,
*conv2d_transpose_2/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?}
?	
A__inference_VAE_layer_call_and_return_conditional_losses_86804801
x
featureextractor_86804671
featureextractor_86804673
featureextractor_86804675
featureextractor_86804677
featureextractor_86804679
featureextractor_86804681
featureextractor_86804683
featureextractor_86804685
dense_1_86804688
dense_1_86804690
dense_2_86804693
dense_2_86804695
dense_3_86804715
dense_3_86804717
dense_4_86804734
dense_4_86804736
dense_5_86804743
dense_5_86804745
dense_6_86804748
dense_6_86804750
decoder_86804772
decoder_86804774
decoder_86804776
decoder_86804778
decoder_86804780
decoder_86804782
decoder_86804784
decoder_86804786
decoder_86804788
decoder_86804790
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86804671featureextractor_86804673featureextractor_86804675featureextractor_86804677featureextractor_86804679featureextractor_86804681featureextractor_86804683featureextractor_86804685*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_868034842*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall1FeatureExtractor/StatefulPartitionedCall:output:0dense_1_86804688dense_1_86804690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_868042522!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall1FeatureExtractor/StatefulPartitionedCall:output:0dense_2_86804693dense_2_86804695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_868042782!
dense_2/StatefulPartitionedCall?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y?
truedivRealDiv(dense_2/StatefulPartitionedCall:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulx
addAddV2(dense_1/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/StatefulPartitionedCallStatefulPartitionedCall1FeatureExtractor/StatefulPartitionedCall:output:0dense_3_86804715dense_3_86804717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_868043212!
dense_3/StatefulPartitionedCallj
Shape_1Shape(dense_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1~
add_3AddV2(dense_3/StatefulPartitionedCall:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?+?2
truediv_1/yt
	truediv_1RealDiv	add_3:z:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????
2
	truediv_1^
SoftmaxSoftmaxtruediv_1:z:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
dense_4/StatefulPartitionedCallStatefulPartitionedCallSoftmax:softmax:0dense_4_86804734dense_4_86804736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_868043612!
dense_4/StatefulPartitionedCallW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?A??2	
mul_1/x?
mul_1Mulmul_1/x:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/StatefulPartitionedCallStatefulPartitionedCall	mul_2:z:0dense_5_86804743dense_5_86804745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_868043912!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall	mul_2:z:0dense_6_86804748dense_6_86804750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_868044172!
dense_6/StatefulPartitionedCallC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/y?
	truediv_2RealDiv(dense_6/StatefulPartitionedCall:output:0truediv_2/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_2V
Exp_1Exptruediv_2:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3~
add_4AddV2(dense_5/StatefulPartitionedCall:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86804772decoder_86804774decoder_86804776decoder_86804778decoder_86804780decoder_86804782decoder_86804784decoder_86804786decoder_86804788decoder_86804790*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_868041062!
Decoder/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity(dense_2/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity(dense_3/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity(dense_5/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity(dense_6/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
??
?
__inference_call_86806977
x
featureextractor_86806752
featureextractor_86806754
featureextractor_86806756
featureextractor_86806758
featureextractor_86806760
featureextractor_86806762
featureextractor_86806764
featureextractor_86806766*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86806948
decoder_86806950
decoder_86806952
decoder_86806954
decoder_86806956
decoder_86806958
decoder_86806960
decoder_86806962
decoder_86806964
decoder_86806966
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86806752featureextractor_86806754featureextractor_86806756featureextractor_86806758featureextractor_86806760featureextractor_86806762featureextractor_86806764featureextractor_86806766*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868067512*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yt
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*
_output_shapes

:2	
truedivG
ExpExptruediv:z:0*
T0*
_output_shapes

:2
ExpV
mulMulExp:y:0random_normal:z:0*
T0*
_output_shapes

:2
mul_
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*
_output_shapes

:2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
2
dense_3/BiasAddc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*
_output_shapes

:
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yy
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*
_output_shapes

:
2
add_1E
LogLog	add_1:z:0*
T0*
_output_shapes

:
2
LogC
NegNegLog:y:0*
T0*
_output_shapes

:
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/y[
add_2AddV2Neg:y:0add_2/y:output:0*
T0*
_output_shapes

:
2
add_2I
Log_1Log	add_2:z:0*
T0*
_output_shapes

:
2
Log_1I
Neg_1Neg	Log_1:y:0*
T0*
_output_shapes

:
2
Neg_1e
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*
_output_shapes

:
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicesv
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
MaxY
EqualEqual	add_3:z:0Max:output:0*
T0*
_output_shapes

:
2
EqualW
CastCast	Equal:z:0*

DstT0*

SrcT0
*
_output_shapes

:
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?|??2	
mul_1/xk
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	?2
mul_1R
SigmoidSigmoid	mul_1:z:0*
T0*
_output_shapes
:	?2	
Sigmoid
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*
_output_shapes
:	?2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/BiasAddk
Shape_2Const*
_output_shapes
:*
dtype0*%
valueB"            2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes

:*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*
_output_shapes

:2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes

:2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yz
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*
_output_shapes

:2
	truediv_1M
Exp_1Exptruediv_1:z:0*
T0*
_output_shapes

:2
Exp_1^
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*
_output_shapes

:2
mul_3e
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*
_output_shapes

:2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axisx
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*
_output_shapes

:2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86806948decoder_86806950decoder_86806952decoder_86806954decoder_86806956decoder_86806958decoder_86806960decoder_86806962decoder_86806964decoder_86806966*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868069472!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*&
_output_shapes
:2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
?

*__inference_dense_4_layer_call_fn_86807924

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_868043612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

*__inference_dense_1_layer_call_fn_86807867

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_868042522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_86803123

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
Қ
?	
__inference_call_86808452
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:

_user_specified_namex
?
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_86803713

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?

E__inference_Decoder_layer_call_and_return_conditional_losses_86808138
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeShape'sequential_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape'sequential_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape'sequential_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_namex
?

?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803460
input_1
sequential_86803442
sequential_86803444
sequential_86803446
sequential_86803448
sequential_86803450
sequential_86803452
sequential_86803454
sequential_86803456
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_86803442sequential_86803444sequential_86803446sequential_86803448sequential_86803450sequential_86803452sequential_86803454sequential_86803456*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033562$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
E__inference_re_lu_4_layer_call_and_return_conditional_losses_86803792

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_reshape_layer_call_fn_86809256

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_868037742
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
Қ
?	
__inference_call_86807470
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:@

_user_specified_namex
??
?
__inference_call_86806313
x
featureextractor_86806177
featureextractor_86806179
featureextractor_86806181
featureextractor_86806183
featureextractor_86806185
featureextractor_86806187
featureextractor_86806189
featureextractor_86806191*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86806284
decoder_86806286
decoder_86806288
decoder_86806290
decoder_86806292
decoder_86806294
decoder_86806296
decoder_86806298
decoder_86806300
decoder_86806302
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86806177featureextractor_86806179featureextractor_86806181featureextractor_86806183featureextractor_86806185featureextractor_86806187featureextractor_86806189featureextractor_86806191*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868052142*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?+?2
truediv_1/yt
	truediv_1RealDiv	add_3:z:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????
2
	truediv_1^
SoftmaxSoftmaxtruediv_1:z:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulSoftmax:softmax:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?h??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/y?
	truediv_2RealDivdense_6/BiasAdd:output:0truediv_2/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_2V
Exp_1Exptruediv_2:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86806284decoder_86806286decoder_86806288decoder_86806290decoder_86806292decoder_86806294decoder_86806296decoder_86806298decoder_86806300decoder_86806302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868054092!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?

*__inference_dense_8_layer_call_fn_86809227

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_868037312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
__inference_call_86802993
x
featureextractor_86802768
featureextractor_86802770
featureextractor_86802772
featureextractor_86802774
featureextractor_86802776
featureextractor_86802778
featureextractor_86802780
featureextractor_86802782*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86802964
decoder_86802966
decoder_86802968
decoder_86802970
decoder_86802972
decoder_86802974
decoder_86802976
decoder_86802978
decoder_86802980
decoder_86802982
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86802768featureextractor_86802770featureextractor_86802772featureextractor_86802774featureextractor_86802776featureextractor_86802778featureextractor_86802780featureextractor_86802782*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868027672*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indices
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Maxb
EqualEqual	add_3:z:0Max:output:0*
T0*'
_output_shapes
:?????????
2
Equal`
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?-??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/y?
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_1V
Exp_1Exptruediv_1:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86802964decoder_86802966decoder_86802968decoder_86802970decoder_86802972decoder_86802974decoder_86802976decoder_86802978decoder_86802980decoder_86802982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868029632!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?2
?
__inference_call_86807013
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	@?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	@?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	@?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:@

_user_specified_namex
?
?
#__inference__wrapped_model_86803070
input_1
vae_86802994
vae_86802996
vae_86802998
vae_86803000
vae_86803002
vae_86803004
vae_86803006
vae_86803008
vae_86803010
vae_86803012
vae_86803014
vae_86803016
vae_86803018
vae_86803020
vae_86803022
vae_86803024
vae_86803026
vae_86803028
vae_86803030
vae_86803032
vae_86803034
vae_86803036
vae_86803038
vae_86803040
vae_86803042
vae_86803044
vae_86803046
vae_86803048
vae_86803050
vae_86803052
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??VAE/StatefulPartitionedCall?
VAE/StatefulPartitionedCallStatefulPartitionedCallinput_1vae_86802994vae_86802996vae_86802998vae_86803000vae_86803002vae_86803004vae_86803006vae_86803008vae_86803010vae_86803012vae_86803014vae_86803016vae_86803018vae_86803020vae_86803022vae_86803024vae_86803026vae_86803028vae_86803030vae_86803032vae_86803034vae_86803036vae_86803038vae_86803040vae_86803042vae_86803044vae_86803046vae_86803048vae_86803050vae_86803052**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:?????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868029932
VAE/StatefulPartitionedCall?
IdentityIdentity$VAE/StatefulPartitionedCall:output:0^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity$VAE/StatefulPartitionedCall:output:1^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity$VAE/StatefulPartitionedCall:output:2^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity$VAE/StatefulPartitionedCall:output:3^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity$VAE/StatefulPartitionedCall:output:4^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity$VAE/StatefulPartitionedCall:output:5^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity$VAE/StatefulPartitionedCall:output:6^VAE/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity$VAE/StatefulPartitionedCall:output:7^VAE/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2:
VAE/StatefulPartitionedCallVAE/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
__inference_call_86807239
x
featureextractor_86807014
featureextractor_86807016
featureextractor_86807018
featureextractor_86807020
featureextractor_86807022
featureextractor_86807024
featureextractor_86807026
featureextractor_86807028*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86807210
decoder_86807212
decoder_86807214
decoder_86807216
decoder_86807218
decoder_86807220
decoder_86807222
decoder_86807224
decoder_86807226
decoder_86807228
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86807014featureextractor_86807016featureextractor_86807018featureextractor_86807020featureextractor_86807022featureextractor_86807024featureextractor_86807026featureextractor_86807028*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	@?**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868070132*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_2/BiasAddg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@            2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:@2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yt
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*
_output_shapes

:@2	
truedivG
ExpExptruediv:z:0*
T0*
_output_shapes

:@2
ExpV
mulMulExp:y:0random_normal:z:0*
T0*
_output_shapes

:@2
mul_
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*
_output_shapes

:@2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
dense_3/BiasAddc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@   
   2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*
_output_shapes

:@
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yy
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*
_output_shapes

:@
2
add_1E
LogLog	add_1:z:0*
T0*
_output_shapes

:@
2
LogC
NegNegLog:y:0*
T0*
_output_shapes

:@
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/y[
add_2AddV2Neg:y:0add_2/y:output:0*
T0*
_output_shapes

:@
2
add_2I
Log_1Log	add_2:z:0*
T0*
_output_shapes

:@
2
Log_1I
Neg_1Neg	Log_1:y:0*
T0*
_output_shapes

:@
2
Neg_1e
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*
_output_shapes

:@
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicesv
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
MaxY
EqualEqual	add_3:z:0Max:output:0*
T0*
_output_shapes

:@
2
EqualW
CastCast	Equal:z:0*

DstT0*

SrcT0
*
_output_shapes

:@
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???2	
mul_1/xk
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
mul_1R
SigmoidSigmoid	mul_1:z:0*
T0*
_output_shapes
:	@?2	
Sigmoid
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*
_output_shapes
:	@?2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_6/BiasAddk
Shape_2Const*
_output_shapes
:*
dtype0*%
valueB"@            2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes

:@*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*
_output_shapes

:@2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes

:@2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yz
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*
_output_shapes

:@2
	truediv_1M
Exp_1Exptruediv_1:z:0*
T0*
_output_shapes

:@2
Exp_1^
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*
_output_shapes

:@2
mul_3e
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*
_output_shapes

:@2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axisx
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*
_output_shapes

:@2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86807210decoder_86807212decoder_86807214decoder_86807216decoder_86807218decoder_86807220decoder_86807222decoder_86807224decoder_86807226decoder_86807228*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868072092!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*&
_output_shapes
:@2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:@::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:I E
&
_output_shapes
:@

_user_specified_namex
?	
?
*__inference_Decoder_layer_call_fn_86804179
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_868041562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_86803162

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_86809091

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_8_layer_call_and_return_conditional_losses_86809218

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_re_lu_2_layer_call_fn_86809208

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_868037132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?3
?
__inference_call_86807848
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_86803215

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_86803375
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?	
?
E__inference_dense_7_layer_call_and_return_conditional_losses_86803692

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_1_layer_call_fn_86809052

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868039542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
H__inference_sequential_layer_call_and_return_conditional_losses_86803245
conv2d_input
conv2d_86803095
conv2d_86803097
conv2d_1_86803134
conv2d_1_86803136
conv2d_2_86803173
conv2d_2_86803175
dense_86803226
dense_86803228
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_86803095conv2d_86803097*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_868030842 
conv2d/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_868031052
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_86803134conv2d_1_86803136*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_868031232"
 conv2d_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_868031442
re_lu_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_86803173conv2d_2_86803175*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_868031622"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_868031832
leaky_re_lu/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_868031972
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_86803226dense_86803228*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_868032152
dense/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_868032362
leaky_re_lu_1/PartitionedCall?
IdentityIdentity&leaky_re_lu_1/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_86803183

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_re_lu_1_layer_call_fn_86809110

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_868031442
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_86804321

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_1_layer_call_fn_86803633

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_868036232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_layer_call_fn_86809139

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_868031832
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Қ
?	
__inference_call_86808364
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:

_user_specified_namex
?
a
E__inference_reshape_layer_call_and_return_conditional_losses_86803774

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
__inference_call_86807275
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	@?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	@?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	@?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:@

_user_specified_namex
?.
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803858
dense_7_input
dense_7_86803827
dense_7_86803829
dense_8_86803833
dense_8_86803835
conv2d_transpose_86803840
conv2d_transpose_86803842
conv2d_transpose_1_86803846
conv2d_transpose_1_86803848
conv2d_transpose_2_86803852
conv2d_transpose_2_86803854
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_86803827dense_7_86803829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_868036922!
dense_7/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_868037132
re_lu_2/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_8_86803833dense_8_86803835*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_868037312!
dense_8/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_868037522
re_lu_3/PartitionedCall?
reshape/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_868037742
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_86803840conv2d_transpose_86803842*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_868035792*
(conv2d_transpose/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_868037922
re_lu_4/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_transpose_1_86803846conv2d_transpose_1_86803848*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_868036232,
*conv2d_transpose_1/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_868038102
re_lu_5/PartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_transpose_2_86803852conv2d_transpose_2_86803854*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_868036682,
*conv2d_transpose_2/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_7_input
?	
?
/__inference_sequential_1_layer_call_fn_86803918
dense_7_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868038952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_7_input
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_86803197

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_reshape_layer_call_and_return_conditional_losses_86809251

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_86809145

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?4
?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86807534
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
E__inference_dense_7_layer_call_and_return_conditional_losses_86809189

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_layer_call_fn_86809071

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_868030842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_Decoder_layer_call_and_return_conditional_losses_86804053
input_1
sequential_1_86804031
sequential_1_86804033
sequential_1_86804035
sequential_1_86804037
sequential_1_86804039
sequential_1_86804041
sequential_1_86804043
sequential_1_86804045
sequential_1_86804047
sequential_1_86804049
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_86804031sequential_1_86804033sequential_1_86804035sequential_1_86804037sequential_1_86804039sequential_1_86804041sequential_1_86804043sequential_1_86804045sequential_1_86804047sequential_1_86804049*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868038952&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

*__inference_dense_6_layer_call_fn_86807962

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_868044172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_86809160

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_VAE_layer_call_and_return_conditional_losses_86805579
x
featureextractor_86805442
featureextractor_86805444
featureextractor_86805446
featureextractor_86805448
featureextractor_86805450
featureextractor_86805452
featureextractor_86805454
featureextractor_86805456*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86805550
decoder_86805552
decoder_86805554
decoder_86805556
decoder_86805558
decoder_86805560
decoder_86805562
decoder_86805564
decoder_86805566
decoder_86805568
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86805442featureextractor_86805444featureextractor_86805446featureextractor_86805448featureextractor_86805450featureextractor_86805452featureextractor_86805454featureextractor_86805456*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868027672*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indices
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Maxb
EqualEqual	add_3:z:0Max:output:0*
T0*'
_output_shapes
:?????????
2
Equal`
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *2U??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/y?
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_1V
Exp_1Exptruediv_1:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86805550decoder_86805552decoder_86805554decoder_86805556decoder_86805558decoder_86805560decoder_86805562decoder_86805564decoder_86805566decoder_86805568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868029632!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?.
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803824
dense_7_input
dense_7_86803703
dense_7_86803705
dense_8_86803742
dense_8_86803744
conv2d_transpose_86803782
conv2d_transpose_86803784
conv2d_transpose_1_86803800
conv2d_transpose_1_86803802
conv2d_transpose_2_86803818
conv2d_transpose_2_86803820
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_86803703dense_7_86803705*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_868036922!
dense_7/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_868037132
re_lu_2/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_8_86803742dense_8_86803744*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_868037312!
dense_8/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_868037522
re_lu_3/PartitionedCall?
reshape/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_868037742
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_86803782conv2d_transpose_86803784*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_868035792*
(conv2d_transpose/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_868037922
re_lu_4/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_transpose_1_86803800conv2d_transpose_1_86803802*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_868036232,
*conv2d_transpose_1/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_868038102
re_lu_5/PartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_transpose_2_86803818conv2d_transpose_2_86803820*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_868036682,
*conv2d_transpose_2/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_7_input
??
?
A__inference_VAE_layer_call_and_return_conditional_losses_86805439
x
featureextractor_86805215
featureextractor_86805217
featureextractor_86805219
featureextractor_86805221
featureextractor_86805223
featureextractor_86805225
featureextractor_86805227
featureextractor_86805229*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86805410
decoder_86805412
decoder_86805414
decoder_86805416
decoder_86805418
decoder_86805420
decoder_86805422
decoder_86805424
decoder_86805426
decoder_86805428
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86805215featureextractor_86805217featureextractor_86805219featureextractor_86805221featureextractor_86805223featureextractor_86805225featureextractor_86805227featureextractor_86805229*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868052142*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?+?2
truediv_1/yt
	truediv_1RealDiv	add_3:z:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????
2
	truediv_1^
SoftmaxSoftmaxtruediv_1:z:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulSoftmax:softmax:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?N??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/y?
	truediv_2RealDivdense_6/BiasAdd:output:0truediv_2/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_2V
Exp_1Exptruediv_2:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86805410decoder_86805412decoder_86805414decoder_86805416decoder_86805418decoder_86805420decoder_86805422decoder_86805424decoder_86805426decoder_86805428*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868054092!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?4
?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86807568
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
/__inference_sequential_1_layer_call_fn_86803977
dense_7_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868039542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_7_input
?#
?
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_86803579

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?$
?
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86803668

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_Decoder_layer_call_and_return_conditional_losses_86804078
input_1
sequential_1_86804056
sequential_1_86804058
sequential_1_86804060
sequential_1_86804062
sequential_1_86804064
sequential_1_86804066
sequential_1_86804068
sequential_1_86804070
sequential_1_86804072
sequential_1_86804074
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_86804056sequential_1_86804058sequential_1_86804060sequential_1_86804062sequential_1_86804064sequential_1_86804066sequential_1_86804068sequential_1_86804070sequential_1_86804072sequential_1_86804074*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868039542&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
__inference_call_86806453
x
featureextractor_86806316
featureextractor_86806318
featureextractor_86806320
featureextractor_86806322
featureextractor_86806324
featureextractor_86806326
featureextractor_86806328
featureextractor_86806330*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86806424
decoder_86806426
decoder_86806428
decoder_86806430
decoder_86806432
decoder_86806434
decoder_86806436
decoder_86806438
decoder_86806440
decoder_86806442
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86806316featureextractor_86806318featureextractor_86806320featureextractor_86806322featureextractor_86806324featureextractor_86806326featureextractor_86806328featureextractor_86806330*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868027672*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indices
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Maxb
EqualEqual	add_3:z:0Max:output:0*
T0*'
_output_shapes
:?????????
2
Equal`
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *io??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/y?
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_1V
Exp_1Exptruediv_1:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86806424decoder_86806426decoder_86806428decoder_86806430decoder_86806432decoder_86806434decoder_86806436decoder_86806438decoder_86806440decoder_86806442*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868029632!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
3__inference_FeatureExtractor_layer_call_fn_86803545
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_868035262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_86804278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_re_lu_4_layer_call_fn_86809266

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_868037922
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_86805178
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:?????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *,
f'R%
#__inference__wrapped_model_868030702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

*__inference_dense_2_layer_call_fn_86807886

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_868042782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_5_layer_call_and_return_conditional_losses_86809271

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Қ
?	
__inference_call_86808716
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:@

_user_specified_namex
??
?
__inference_call_86806715
x
featureextractor_86806490
featureextractor_86806492
featureextractor_86806494
featureextractor_86806496
featureextractor_86806498
featureextractor_86806500
featureextractor_86806502
featureextractor_86806504*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86806686
decoder_86806688
decoder_86806690
decoder_86806692
decoder_86806694
decoder_86806696
decoder_86806698
decoder_86806700
decoder_86806702
decoder_86806704
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86806490featureextractor_86806492featureextractor_86806494featureextractor_86806496featureextractor_86806498featureextractor_86806500featureextractor_86806502featureextractor_86806504*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868064892*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yt
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*
_output_shapes

:2	
truedivG
ExpExptruediv:z:0*
T0*
_output_shapes

:2
ExpV
mulMulExp:y:0random_normal:z:0*
T0*
_output_shapes

:2
mul_
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*
_output_shapes

:2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
2
dense_3/BiasAddc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*
_output_shapes

:
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yy
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*
_output_shapes

:
2
add_1E
LogLog	add_1:z:0*
T0*
_output_shapes

:
2
LogC
NegNegLog:y:0*
T0*
_output_shapes

:
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/y[
add_2AddV2Neg:y:0add_2/y:output:0*
T0*
_output_shapes

:
2
add_2I
Log_1Log	add_2:z:0*
T0*
_output_shapes

:
2
Log_1I
Neg_1Neg	Log_1:y:0*
T0*
_output_shapes

:
2
Neg_1e
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*
_output_shapes

:
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicesv
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
MaxY
EqualEqual	add_3:z:0Max:output:0*
T0*
_output_shapes

:
2
EqualW
CastCast	Equal:z:0*

DstT0*

SrcT0
*
_output_shapes

:
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?u??2	
mul_1/xk
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	?2
mul_1R
SigmoidSigmoid	mul_1:z:0*
T0*
_output_shapes
:	?2	
Sigmoid
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*
_output_shapes
:	?2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/BiasAddk
Shape_2Const*
_output_shapes
:*
dtype0*%
valueB"            2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes

:*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*
_output_shapes

:2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes

:2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yz
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*
_output_shapes

:2
	truediv_1M
Exp_1Exptruediv_1:z:0*
T0*
_output_shapes

:2
Exp_1^
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*
_output_shapes

:2
mul_3e
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*
_output_shapes

:2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axisx
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*
_output_shapes

:2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86806686decoder_86806688decoder_86806690decoder_86806692decoder_86806694decoder_86806696decoder_86806698decoder_86806700decoder_86806702decoder_86806704*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868066852!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*&
_output_shapes
:2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
?
?
E__inference_Decoder_layer_call_and_return_conditional_losses_86804156
x
sequential_1_86804134
sequential_1_86804136
sequential_1_86804138
sequential_1_86804140
sequential_1_86804142
sequential_1_86804144
sequential_1_86804146
sequential_1_86804148
sequential_1_86804150
sequential_1_86804152
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallxsequential_1_86804134sequential_1_86804136sequential_1_86804138sequential_1_86804140sequential_1_86804142sequential_1_86804144sequential_1_86804146sequential_1_86804148sequential_1_86804150sequential_1_86804152*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868039542&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?	
?
*__inference_Decoder_layer_call_fn_86808163
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_868041062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?
?
&__inference_VAE_layer_call_fn_86805737
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:+???????????????????????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *J
fERC
A__inference_VAE_layer_call_and_return_conditional_losses_868050142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
a
E__inference_re_lu_3_layer_call_and_return_conditional_losses_86809232

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
H__inference_sequential_layer_call_and_return_conditional_losses_86808784

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdds

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAddy
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
re_lu_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2
leaky_re_lu/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2
leaky_re_lu_1/LeakyRelu?
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?2
?
__inference_call_86806751
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
?

?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803526
x
sequential_86803508
sequential_86803510
sequential_86803512
sequential_86803514
sequential_86803516
sequential_86803518
sequential_86803520
sequential_86803522
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_86803508sequential_86803510sequential_86803512sequential_86803514sequential_86803516sequential_86803518sequential_86803520sequential_86803522*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033562$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?#
?
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_86803623

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
D
(__inference_re_lu_layer_call_fn_86809081

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_868031052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_86807858

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?

E__inference_Decoder_layer_call_and_return_conditional_losses_86808050
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeShape'sequential_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape'sequential_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape'sequential_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_namex
??
?
A__inference_VAE_layer_call_and_return_conditional_losses_86806016
input_1
featureextractor_86805879
featureextractor_86805881
featureextractor_86805883
featureextractor_86805885
featureextractor_86805887
featureextractor_86805889
featureextractor_86805891
featureextractor_86805893*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86805987
decoder_86805989
decoder_86805991
decoder_86805993
decoder_86805995
decoder_86805997
decoder_86805999
decoder_86806001
decoder_86806003
decoder_86806005
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallinput_1featureextractor_86805879featureextractor_86805881featureextractor_86805883featureextractor_86805885featureextractor_86805887featureextractor_86805889featureextractor_86805891featureextractor_86805893*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868027672*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddE
ShapeShapeinput_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indices
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Maxb
EqualEqual	add_3:z:0Max:output:0*
T0*'
_output_shapes
:?????????
2
Equal`
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
Cast?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulCast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *Nb??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddI
Shape_2Shapeinput_1*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/y?
	truediv_1RealDivdense_6/BiasAdd:output:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_1V
Exp_1Exptruediv_1:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86805987decoder_86805989decoder_86805991decoder_86805993decoder_86805995decoder_86805997decoder_86805999decoder_86806001decoder_86806003decoder_86806005*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868029632!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?&
?
H__inference_sequential_layer_call_and_return_conditional_losses_86803306

inputs
conv2d_86803280
conv2d_86803282
conv2d_1_86803286
conv2d_1_86803288
conv2d_2_86803292
conv2d_2_86803294
dense_86803299
dense_86803301
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_86803280conv2d_86803282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_868030842 
conv2d/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_868031052
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_86803286conv2d_1_86803288*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_868031232"
 conv2d_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_868031442
re_lu_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_86803292conv2d_2_86803294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_868031622"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_868031832
leaky_re_lu/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_868031972
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_86803299dense_86803301*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_868032152
dense/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_868032362
leaky_re_lu_1/PartitionedCall?
IdentityIdentity&leaky_re_lu_1/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_layer_call_fn_86809169

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_868032152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_layer_call_and_return_conditional_losses_86809076

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_8_layer_call_and_return_conditional_losses_86803731

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
__inference_call_86807746
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	@?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	@?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	@?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:@

_user_specified_namex
?
?
-__inference_sequential_layer_call_fn_86803325
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?~
?	
A__inference_VAE_layer_call_and_return_conditional_losses_86805014
x
featureextractor_86804883
featureextractor_86804885
featureextractor_86804887
featureextractor_86804889
featureextractor_86804891
featureextractor_86804893
featureextractor_86804895
featureextractor_86804897
dense_1_86804900
dense_1_86804902
dense_2_86804905
dense_2_86804907
dense_3_86804927
dense_3_86804929
dense_4_86804947
dense_4_86804949
dense_5_86804956
dense_5_86804958
dense_6_86804961
dense_6_86804963
decoder_86804985
decoder_86804987
decoder_86804989
decoder_86804991
decoder_86804993
decoder_86804995
decoder_86804997
decoder_86804999
decoder_86805001
decoder_86805003
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86804883featureextractor_86804885featureextractor_86804887featureextractor_86804889featureextractor_86804891featureextractor_86804893featureextractor_86804895featureextractor_86804897*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_868035262*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall1FeatureExtractor/StatefulPartitionedCall:output:0dense_1_86804900dense_1_86804902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_868042522!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall1FeatureExtractor/StatefulPartitionedCall:output:0dense_2_86804905dense_2_86804907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_868042782!
dense_2/StatefulPartitionedCall?
ShapeShapex*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y?
truedivRealDiv(dense_2/StatefulPartitionedCall:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulx
addAddV2(dense_1/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/StatefulPartitionedCallStatefulPartitionedCall1FeatureExtractor/StatefulPartitionedCall:output:0dense_3_86804927dense_3_86804929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_868043212!
dense_3/StatefulPartitionedCallj
Shape_1Shape(dense_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1~
add_3AddV2(dense_3/StatefulPartitionedCall:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indices
MaxMax	add_3:z:0Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Maxb
EqualEqual	add_3:z:0Max:output:0*
T0*'
_output_shapes
:?????????
2
Equal`
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
Cast?
dense_4/StatefulPartitionedCallStatefulPartitionedCallCast:y:0dense_4_86804947dense_4_86804949*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_868043612!
dense_4/StatefulPartitionedCallW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *H??2	
mul_1/x?
mul_1Mulmul_1/x:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/StatefulPartitionedCallStatefulPartitionedCall	mul_2:z:0dense_5_86804956dense_5_86804958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_868043912!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall	mul_2:z:0dense_6_86804961dense_6_86804963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_868044172!
dense_6/StatefulPartitionedCallC
Shape_2Shapex*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/y?
	truediv_1RealDiv(dense_6/StatefulPartitionedCall:output:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_1V
Exp_1Exptruediv_1:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3~
add_4AddV2(dense_5/StatefulPartitionedCall:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86804985decoder_86804987decoder_86804989decoder_86804991decoder_86804993decoder_86804995decoder_86804997decoder_86804999decoder_86805001decoder_86805003*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_868041562!
Decoder/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity(dense_2/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity(dense_3/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity(dense_5/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity(dense_6/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?

*__inference_dense_3_layer_call_fn_86807905

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_868043212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_3_layer_call_and_return_conditional_losses_86803752

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?	
__inference_call_86805409
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeShape'sequential_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape'sequential_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape'sequential_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_namex
??
?
A__inference_VAE_layer_call_and_return_conditional_losses_86805876
input_1
featureextractor_86805740
featureextractor_86805742
featureextractor_86805744
featureextractor_86805746
featureextractor_86805748
featureextractor_86805750
featureextractor_86805752
featureextractor_86805754*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86805847
decoder_86805849
decoder_86805851
decoder_86805853
decoder_86805855
decoder_86805857
decoder_86805859
decoder_86805861
decoder_86805863
decoder_86805865
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallinput_1featureextractor_86805740featureextractor_86805742featureextractor_86805744featureextractor_86805746featureextractor_86805748featureextractor_86805750featureextractor_86805752featureextractor_86805754*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868052142*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddE
ShapeShapeinput_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/y}
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mulh
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddZ
Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*'
_output_shapes
:?????????
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/y?
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*'
_output_shapes
:?????????
2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????
2
LogL
NegNegLog:y:0*
T0*'
_output_shapes
:?????????
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/yd
add_2AddV2Neg:y:0add_2/y:output:0*
T0*'
_output_shapes
:?????????
2
add_2R
Log_1Log	add_2:z:0*
T0*'
_output_shapes
:?????????
2
Log_1R
Neg_1Neg	Log_1:y:0*
T0*'
_output_shapes
:?????????
2
Neg_1n
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*'
_output_shapes
:?????????
2
add_3_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?+?2
truediv_1/yt
	truediv_1RealDiv	add_3:z:0truediv_1/y:output:0*
T0*'
_output_shapes
:?????????
2
	truediv_1^
SoftmaxSoftmaxtruediv_1:z:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulSoftmax:softmax:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?[??2	
mul_1/xt
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mul_1[
SigmoidSigmoid	mul_1:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddI
Shape_2Shapeinput_1*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_1_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/y?
	truediv_2RealDivdense_6/BiasAdd:output:0truediv_2/y:output:0*
T0*'
_output_shapes
:?????????2
	truediv_2V
Exp_1Exptruediv_2:z:0*
T0*'
_output_shapes
:?????????2
Exp_1g
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3n
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86805847decoder_86805849decoder_86805851decoder_86805853decoder_86805855decoder_86805857decoder_86805859decoder_86805861decoder_86805863decoder_86805865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868054092!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?&
?
H__inference_sequential_layer_call_and_return_conditional_losses_86803356

inputs
conv2d_86803330
conv2d_86803332
conv2d_1_86803336
conv2d_1_86803338
conv2d_2_86803342
conv2d_2_86803344
dense_86803349
dense_86803351
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_86803330conv2d_86803332*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_868030842 
conv2d/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_868031052
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_86803336conv2d_1_86803338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_868031232"
 conv2d_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_868031442
re_lu_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_86803342conv2d_2_86803344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_868031622"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_868031832
leaky_re_lu/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_868031972
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_86803349dense_86803351*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_868032152
dense/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_868032362
leaky_re_lu_1/PartitionedCall?
IdentityIdentity&leaky_re_lu_1/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_call_86807500
x
featureextractor_86807276
featureextractor_86807278
featureextractor_86807280
featureextractor_86807282
featureextractor_86807284
featureextractor_86807286
featureextractor_86807288
featureextractor_86807290*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
decoder_86807471
decoder_86807473
decoder_86807475
decoder_86807477
decoder_86807479
decoder_86807481
decoder_86807483
decoder_86807485
decoder_86807487
decoder_86807489
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Decoder/StatefulPartitionedCall?(FeatureExtractor/StatefulPartitionedCall?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
(FeatureExtractor/StatefulPartitionedCallStatefulPartitionedCallxfeatureextractor_86807276featureextractor_86807278featureextractor_86807280featureextractor_86807282featureextractor_86807284featureextractor_86807286featureextractor_86807288featureextractor_86807290*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	@?**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868072752*
(FeatureExtractor/StatefulPartitionedCall?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_2/BiasAddg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@            2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal/shape/1?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:@2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@2
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yt
truedivRealDivdense_2/BiasAdd:output:0truediv/y:output:0*
T0*
_output_shapes

:@2	
truedivG
ExpExptruediv:z:0*
T0*
_output_shapes

:@2
ExpV
mulMulExp:y:0random_normal:z:0*
T0*
_output_shapes

:@2
mul_
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*
_output_shapes

:@2
add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul1FeatureExtractor/StatefulPartitionedCall:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
dense_3/BiasAddc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@   
   2	
Shape_1?
random_uniform/RandomUniformRandomUniformShape_1:output:0*
T0*
_output_shapes

:@
*
dtype02
random_uniform/RandomUniformW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yy
add_1AddV2%random_uniform/RandomUniform:output:0add_1/y:output:0*
T0*
_output_shapes

:@
2
add_1E
LogLog	add_1:z:0*
T0*
_output_shapes

:@
2
LogC
NegNegLog:y:0*
T0*
_output_shapes

:@
2
NegW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_2/y[
add_2AddV2Neg:y:0add_2/y:output:0*
T0*
_output_shapes

:@
2
add_2I
Log_1Log	add_2:z:0*
T0*
_output_shapes

:@
2
Log_1I
Neg_1Neg	Log_1:y:0*
T0*
_output_shapes

:@
2
Neg_1e
add_3AddV2dense_3/BiasAdd:output:0	Neg_1:y:0*
T0*
_output_shapes

:@
2
add_3_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?+?2
truediv_1/yk
	truediv_1RealDiv	add_3:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@
2
	truediv_1U
SoftmaxSoftmaxtruediv_1:z:0*
T0*
_output_shapes

:@
2	
Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulSoftmax:softmax:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
dense_4/BiasAddW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
mul_1/xk
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
mul_1R
SigmoidSigmoid	mul_1:z:0*
T0*
_output_shapes
:	@?2	
Sigmoid
mul_2Mul1FeatureExtractor/StatefulPartitionedCall:output:0Sigmoid:y:0*
T0*
_output_shapes
:	@?2
mul_2?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul	mul_2:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul	mul_2:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_6/BiasAddk
Shape_2Const*
_output_shapes
:*
dtype0*%
valueB"@            2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1t
random_normal_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
random_normal_1/shape/1?
random_normal_1/shapePackstrided_slice_1:output:0 random_normal_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/meanu
random_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal_1/stddev?
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes

:@*
dtype02&
$random_normal_1/RandomStandardNormal?
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0random_normal_1/stddev:output:0*
T0*
_output_shapes

:@2
random_normal_1/mul?
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes

:@2
random_normal_1_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yz
	truediv_2RealDivdense_6/BiasAdd:output:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2M
Exp_1Exptruediv_2:z:0*
T0*
_output_shapes

:@2
Exp_1^
mul_3Mul	Exp_1:y:0random_normal_1:z:0*
T0*
_output_shapes

:@2
mul_3e
add_4AddV2dense_5/BiasAdd:output:0	mul_3:z:0*
T0*
_output_shapes

:@2
add_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axisx
concatConcatV2add:z:0	add_4:z:0concat/axis:output:0*
N*
T0*
_output_shapes

:@2
concat?
Decoder/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0decoder_86807471decoder_86807473decoder_86807475decoder_86807477decoder_86807479decoder_86807481decoder_86807483decoder_86807485decoder_86807487decoder_86807489*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *"
fR
__inference_call_868074702!
Decoder/StatefulPartitionedCall?
IdentityIdentitydense_1/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identityadd:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_2?

Identity_3Identitydense_3/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@
2

Identity_3?

Identity_4Identitydense_5/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_4?

Identity_5Identitydense_6/BiasAdd:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_5?

Identity_6Identity	add_4:z:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_6?

Identity_7Identity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall)^FeatureExtractor/StatefulPartitionedCall^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*&
_output_shapes
:@2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:@::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2T
(FeatureExtractor/StatefulPartitionedCall(FeatureExtractor/StatefulPartitionedCall2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:I E
&
_output_shapes
:@

_user_specified_namex
?
?
3__inference_FeatureExtractor_layer_call_fn_86807610
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_868035262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?*
?
H__inference_sequential_layer_call_and_return_conditional_losses_86808750

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdds

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAddy
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
re_lu_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2
leaky_re_lu/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2
leaky_re_lu_1/LeakyRelu?
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
__inference_call_86802767
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
5__inference_conv2d_transpose_2_layer_call_fn_86803678

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_868036682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
E__inference_re_lu_4_layer_call_and_return_conditional_losses_86809261

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_86803236

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???=2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_86808805

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_86809120

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_re_lu_3_layer_call_fn_86809237

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_868037522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_86804252

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
__inference_call_86807678
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
Қ
?	
__inference_call_86806685
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:

_user_specified_namex
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_86804361

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?2
?
__inference_call_86807814
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
?2
?
__inference_call_86807712
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	@?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	@?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	@?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:@

_user_specified_namex
Қ
?	
__inference_call_86808628
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:@

_user_specified_namex
?
L
0__inference_leaky_re_lu_1_layer_call_fn_86809179

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_868032362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_86807915

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?C
?
!__inference__traced_save_86809402
file_prefix,
(savev2_u_prior_means_read_readvariableop:
6savev2_u_prior_logvars_before_tanh_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B(u_prior_means/.ATTRIBUTES/VARIABLE_VALUEB6u_prior_logvars_before_tanh/.ATTRIBUTES/VARIABLE_VALUEB.z_mean_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,z_mean_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0z_logvar_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB.z_logvar_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB.h_to_c_logit/kernel/.ATTRIBUTES/VARIABLE_VALUEB,h_to_c_logit/bias/.ATTRIBUTES/VARIABLE_VALUEB.c_to_a_logit/kernel/.ATTRIBUTES/VARIABLE_VALUEB,c_to_a_logit/bias/.ATTRIBUTES/VARIABLE_VALUEB.u_mean_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,u_mean_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0u_logvar_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB.u_logvar_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_u_prior_means_read_readvariableop6savev2_u_prior_logvars_before_tanh_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
:
:	?::	?::	?
:
:	
?:?:	?::	?:: : : @:@:@@:@:
??:?:	?:?:
??:?: @: :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
:$ 

_output_shapes

:
:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?
: 

_output_shapes
:
:%	!

_output_shapes
:	
?:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: :  

_output_shapes
::!

_output_shapes
: 
?3
?
__inference_call_86805214
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86808914

inputs*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddq
re_lu_2/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
re_lu_2/Relu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulre_lu_2/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAddq
re_lu_3/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
re_lu_3/Reluh
reshape/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapere_lu_3/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose/BiasAdd?
re_lu_4/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_4/Relu~
conv2d_transpose_1/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
re_lu_5/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_5/Relu~
conv2d_transpose_2/ShapeShapere_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/BiasAdd?
conv2d_transpose_2/SigmoidSigmoid#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/Sigmoid?
IdentityIdentityconv2d_transpose_2/Sigmoid:y:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_5_layer_call_and_return_conditional_losses_86804391

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_86808826

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
__inference_call_86807644
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?

*__inference_dense_7_layer_call_fn_86809198

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_868036922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_re_lu_5_layer_call_fn_86809276

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_868038102
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
E__inference_re_lu_5_layer_call_and_return_conditional_losses_86803810

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Қ
?	
__inference_call_86806947
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:

_user_specified_namex
?

?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803439
input_1
sequential_86803421
sequential_86803423
sequential_86803425
sequential_86803427
sequential_86803429
sequential_86803431
sequential_86803433
sequential_86803435
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_86803421sequential_86803423sequential_86803425sequential_86803427sequential_86803429sequential_86803431sequential_86803433sequential_86803435*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033062$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
Қ
?	
__inference_call_86807209
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@?2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @  2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:@@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@         @   2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*&
_output_shapes
:@ *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@ 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
:@ 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"@             2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:@::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:A =

_output_shapes

:@

_user_specified_namex
?
?
3__inference_FeatureExtractor_layer_call_fn_86807589
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_868034842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
/__inference_sequential_1_layer_call_fn_86809027

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868038952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_Decoder_layer_call_fn_86804129
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_868041062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_86809134

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_86809203

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_VAE_layer_call_fn_86806095
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:+???????????????????????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *J
fERC
A__inference_VAE_layer_call_and_return_conditional_losses_868048012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_dense_5_layer_call_and_return_conditional_losses_86807934

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
H__inference_sequential_layer_call_and_return_conditional_losses_86803274
conv2d_input
conv2d_86803248
conv2d_86803250
conv2d_1_86803254
conv2d_1_86803256
conv2d_2_86803260
conv2d_2_86803262
dense_86803267
dense_86803269
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_86803248conv2d_86803250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_868030842 
conv2d/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_868031052
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_86803254conv2d_1_86803256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_868031232"
 conv2d_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_868031442
re_lu_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_86803260conv2d_2_86803262*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_868031622"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_868031832
leaky_re_lu/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_868031972
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_86803267dense_86803269*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_868032152
dense/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_868032362
leaky_re_lu_1/PartitionedCall?
IdentityIdentity&leaky_re_lu_1/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?.
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803954

inputs
dense_7_86803923
dense_7_86803925
dense_8_86803929
dense_8_86803931
conv2d_transpose_86803936
conv2d_transpose_86803938
conv2d_transpose_1_86803942
conv2d_transpose_1_86803944
conv2d_transpose_2_86803948
conv2d_transpose_2_86803950
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_86803923dense_7_86803925*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_868036922!
dense_7/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_868037132
re_lu_2/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_8_86803929dense_8_86803931*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_868037312!
dense_8/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_868037522
re_lu_3/PartitionedCall?
reshape/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_868037742
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_86803936conv2d_transpose_86803938*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_868035792*
(conv2d_transpose/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_868037922
re_lu_4/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_transpose_1_86803942conv2d_transpose_1_86803944*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_868036232,
*conv2d_transpose_1/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_868038102
re_lu_5/PartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_transpose_2_86803948conv2d_transpose_2_86803950*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_868036682,
*conv2d_transpose_2/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_1_layer_call_fn_86809100

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_868031232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
3__inference_FeatureExtractor_layer_call_fn_86803503
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_868034842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803484
x
sequential_86803466
sequential_86803468
sequential_86803470
sequential_86803472
sequential_86803474
sequential_86803476
sequential_86803478
sequential_86803480
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_86803466sequential_86803468sequential_86803470sequential_86803472sequential_86803474sequential_86803476sequential_86803478sequential_86803480*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_868033062$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?2
?
__inference_call_86807780
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_86807877

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_5_layer_call_fn_86807943

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_868043912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_layer_call_fn_86803589

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_868035792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?2
?
__inference_call_86806489
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2
sequential/conv2d/BiasAdd?
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
: 2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_1/BiasAdd?
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
sequential/re_lu_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
sequential/conv2d_2/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu$sequential/conv2d_2/BiasAdd:output:0*&
_output_shapes
:@*
alpha%???=2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0!sequential/flatten/Const:output:0*
T0*
_output_shapes
:	?2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*
_output_shapes
:	?*
alpha%???=2$
"sequential/leaky_re_lu_1/LeakyRelu?
IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_86807953

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_VAE_layer_call_fn_86806174
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:+???????????????????????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *J
fERC
A__inference_VAE_layer_call_and_return_conditional_losses_868050142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_86803144

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
*__inference_Decoder_layer_call_fn_86808188
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_868041562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_86804417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_86809174

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???=2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_layer_call_and_return_conditional_losses_86803105

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
&__inference_VAE_layer_call_fn_86805658
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????
:?????????:?????????:?????????:+???????????????????????????*@
_read_only_resource_inputs"
 	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *J
fERC
A__inference_VAE_layer_call_and_return_conditional_losses_868048012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity_7"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
??
?	
__inference_call_86802963
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeShape'sequential_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape'sequential_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape'sequential_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_namex
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86809002

inputs*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddq
re_lu_2/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
re_lu_2/Relu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulre_lu_2/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAddq
re_lu_3/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
re_lu_3/Reluh
reshape/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapere_lu_3/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose/BiasAdd?
re_lu_4/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_4/Relu~
conv2d_transpose_1/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
re_lu_5/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_5/Relu~
conv2d_transpose_2/ShapeShapere_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/BiasAdd?
conv2d_transpose_2/SigmoidSigmoid#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/Sigmoid?
IdentityIdentityconv2d_transpose_2/Sigmoid:y:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_layer_call_and_return_conditional_losses_86809062

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_2_layer_call_fn_86809129

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_868031622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_Decoder_layer_call_and_return_conditional_losses_86804106
x
sequential_1_86804084
sequential_1_86804086
sequential_1_86804088
sequential_1_86804090
sequential_1_86804092
sequential_1_86804094
sequential_1_86804096
sequential_1_86804098
sequential_1_86804100
sequential_1_86804102
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallxsequential_1_86804084sequential_1_86804086sequential_1_86804088sequential_1_86804090sequential_1_86804092sequential_1_86804094sequential_1_86804096sequential_1_86804098sequential_1_86804100sequential_1_86804102*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_868038952&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
??
?	
__inference_call_86808276
x7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource7
3sequential_1_dense_8_matmul_readvariableop_resource8
4sequential_1_dense_8_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?*sequential_1/dense_7/MatMul/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?*sequential_1/dense_8/MatMul/ReadVariableOp?
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOp?
sequential_1/dense_7/MatMulMatMulx2sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/MatMul?
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOp?
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_7/BiasAdd?
sequential_1/re_lu_2/ReluRelu%sequential_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_2/Relu?
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp?
sequential_1/dense_8/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/MatMul?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp?
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_8/BiasAdd?
sequential_1/re_lu_3/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/re_lu_3/Relu?
sequential_1/reshape/ShapeShape'sequential_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/reshape/Reshape/shape/3?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/re_lu_3/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/reshape/Reshape?
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/1?
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/2?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2'
%sequential_1/conv2d_transpose/BiasAdd?
sequential_1/re_lu_4/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_4/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape'sequential_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/1?
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/stack/2?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
sequential_1/re_lu_5/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/re_lu_5/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape'sequential_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
'sequential_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/1?
'sequential_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/2?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:00sequential_1/conv2d_transpose_2/stack/1:output:00sequential_1/conv2d_transpose_2/stack/2:output:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0'sequential_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/BiasAdd?
'sequential_1/conv2d_transpose_2/SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_1/conv2d_transpose_2/Sigmoid?
IdentityIdentity+sequential_1/conv2d_transpose_2/Sigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_namex
?
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_86809105

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_flatten_layer_call_fn_86809150

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_868031972
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_layer_call_and_return_conditional_losses_86803084

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_86807896

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????<
output_30
StatefulPartitionedCall:2?????????<
output_40
StatefulPartitionedCall:3?????????
<
output_50
StatefulPartitionedCall:4?????????<
output_60
StatefulPartitionedCall:5?????????<
output_70
StatefulPartitionedCall:6?????????D
output_88
StatefulPartitionedCall:7?????????tensorflow/serving/predict:??
?
feature_extractor
z_mean_layer
z_logvar_layer
h_to_c_logit
c_to_a_logit
u_mean_layer
u_logvar_layer
decoder
	u_prior_means

u_prior_logvars_before_tanh
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__
	?call"?
_tf_keras_model?{"class_name": "VAE", "name": "VAE", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VAE"}}
?
	nChannels
net
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__
	?call"?
_tf_keras_model?{"class_name": "FeatureExtractor", "name": "FeatureExtractor", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "FeatureExtractor"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
:	nChannels
;net
<regularization_losses
=trainable_variables
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__
	?call"?
_tf_keras_model?{"class_name": "Decoder", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Decoder"}}
:
2u_prior_means
-:+
2u_prior_logvars_before_tanh
 "
trackable_list_wrapper
?
@0
A1
B2
C3
D4
E5
F6
G7
8
9
10
11
"12
#13
(14
)15
.16
/17
418
519
H20
I21
J22
K23
L24
M25
N26
O27
P28
Q29
	30

31"
trackable_list_wrapper
?
@0
A1
B2
C3
D4
E5
F6
G7
8
9
10
11
"12
#13
(14
)15
.16
/17
418
519
H20
I21
J22
K23
L24
M25
N26
O27
P28
Q29
	30

31"
trackable_list_wrapper
?
regularization_losses
trainable_variables
Rmetrics
Slayer_metrics
Tnon_trainable_variables
	variables
Ulayer_regularization_losses

Vlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
?<
Wlayer_with_weights-0
Wlayer-0
Xlayer-1
Ylayer_with_weights-1
Ylayer-2
Zlayer-3
[layer_with_weights-2
[layer-4
\layer-5
]layer-6
^layer_with_weights-3
^layer-7
_layer-8
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?9
_tf_keras_sequential?9{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}]}}}
 "
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
?
regularization_losses
trainable_variables
dmetrics
elayer_metrics
fnon_trainable_variables
	variables
glayer_regularization_losses

hlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
imetrics
jlayer_metrics
knon_trainable_variables
	variables
llayer_regularization_losses

mlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
nmetrics
olayer_metrics
pnon_trainable_variables
 	variables
qlayer_regularization_losses

rlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?
2dense_3/kernel
:
2dense_3/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
$regularization_losses
%trainable_variables
smetrics
tlayer_metrics
unon_trainable_variables
&	variables
vlayer_regularization_losses

wlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	
?2dense_4/kernel
:?2dense_4/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
*regularization_losses
+trainable_variables
xmetrics
ylayer_metrics
znon_trainable_variables
,	variables
{layer_regularization_losses

|layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
0regularization_losses
1trainable_variables
}metrics
~layer_metrics
non_trainable_variables
2	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_6/kernel
:2dense_6/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
6regularization_losses
7trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
8	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?F
?layer_with_weights-0
?layer-0
?layer-1
?layer_with_weights-1
?layer-2
?layer-3
?layer-4
?layer_with_weights-2
?layer-5
?layer-6
?layer_with_weights-3
?layer-7
?layer-8
?layer_with_weights-4
?layer-9
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?C
_tf_keras_sequential?C{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_7_input"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3136, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 64]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_7_input"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3136, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 64]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}}
 "
trackable_list_wrapper
f
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9"
trackable_list_wrapper
f
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9"
trackable_list_wrapper
?
<regularization_losses
=trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
>	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
):' @2conv2d_1/kernel
:@2conv2d_1/bias
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
 :
??2dense/kernel
:?2
dense/bias
!:	?2dense_7/kernel
:?2dense_7/bias
": 
??2dense_8/kernel
:?2dense_8/bias
1:/ @2conv2d_transpose/kernel
#:! 2conv2d_transpose/bias
3:1  2conv2d_transpose_1/kernel
%:# 2conv2d_transpose_1/bias
3:1 2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?	

@kernel
Abias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	

Bkernel
Cbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	

Dkernel
Ebias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Fkernel
Gbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
 "
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
?
`regularization_losses
atrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
b	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Hkernel
Ibias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?

Jkernel
Kbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3136, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 64]}}}
?


Lkernel
Mbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


Nkernel
Obias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


Pkernel
Qbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
 "
trackable_list_wrapper
f
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9"
trackable_list_wrapper
f
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
W0
X1
Y2
Z3
[4
\5
]6
^7
_8"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
p
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
#__inference__wrapped_model_86803070?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
A__inference_VAE_layer_call_and_return_conditional_losses_86805439
A__inference_VAE_layer_call_and_return_conditional_losses_86806016
A__inference_VAE_layer_call_and_return_conditional_losses_86805579
A__inference_VAE_layer_call_and_return_conditional_losses_86805876?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_VAE_layer_call_fn_86805737
&__inference_VAE_layer_call_fn_86805658
&__inference_VAE_layer_call_fn_86806095
&__inference_VAE_layer_call_fn_86806174?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_call_86806313
__inference_call_86806453
__inference_call_86807239
__inference_call_86806715
__inference_call_86807500
__inference_call_86806977?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803439
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86807568
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803460
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86807534?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_FeatureExtractor_layer_call_fn_86807589
3__inference_FeatureExtractor_layer_call_fn_86803503
3__inference_FeatureExtractor_layer_call_fn_86807610
3__inference_FeatureExtractor_layer_call_fn_86803545?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_call_86807644
__inference_call_86807678
__inference_call_86807848
__inference_call_86807712
__inference_call_86807814
__inference_call_86807746
__inference_call_86807780?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_86807858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_86807867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_2_layer_call_and_return_conditional_losses_86807877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_2_layer_call_fn_86807886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_3_layer_call_and_return_conditional_losses_86807896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_3_layer_call_fn_86807905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_4_layer_call_and_return_conditional_losses_86807915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_4_layer_call_fn_86807924?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_5_layer_call_and_return_conditional_losses_86807934?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_5_layer_call_fn_86807943?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_6_layer_call_and_return_conditional_losses_86807953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_6_layer_call_fn_86807962?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_Decoder_layer_call_and_return_conditional_losses_86808138
E__inference_Decoder_layer_call_and_return_conditional_losses_86804053
E__inference_Decoder_layer_call_and_return_conditional_losses_86808050
E__inference_Decoder_layer_call_and_return_conditional_losses_86804078?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_Decoder_layer_call_fn_86808188
*__inference_Decoder_layer_call_fn_86808163
*__inference_Decoder_layer_call_fn_86804129
*__inference_Decoder_layer_call_fn_86804179?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_call_86808276
__inference_call_86808364
__inference_call_86808452
__inference_call_86808628
__inference_call_86808716
__inference_call_86808540?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_86805178input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_sequential_layer_call_and_return_conditional_losses_86803274
H__inference_sequential_layer_call_and_return_conditional_losses_86808750
H__inference_sequential_layer_call_and_return_conditional_losses_86803245
H__inference_sequential_layer_call_and_return_conditional_losses_86808784?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_layer_call_fn_86808826
-__inference_sequential_layer_call_fn_86808805
-__inference_sequential_layer_call_fn_86803375
-__inference_sequential_layer_call_fn_86803325?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803824
J__inference_sequential_1_layer_call_and_return_conditional_losses_86808914
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803858
J__inference_sequential_1_layer_call_and_return_conditional_losses_86809002?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_1_layer_call_fn_86809027
/__inference_sequential_1_layer_call_fn_86809052
/__inference_sequential_1_layer_call_fn_86803977
/__inference_sequential_1_layer_call_fn_86803918?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_layer_call_and_return_conditional_losses_86809062?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_layer_call_fn_86809071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_re_lu_layer_call_and_return_conditional_losses_86809076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_re_lu_layer_call_fn_86809081?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_86809091?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_1_layer_call_fn_86809100?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_1_layer_call_and_return_conditional_losses_86809105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_re_lu_1_layer_call_fn_86809110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_86809120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_2_layer_call_fn_86809129?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_86809134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_layer_call_fn_86809139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_layer_call_and_return_conditional_losses_86809145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_layer_call_fn_86809150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_86809160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_86809169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_86809174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_1_layer_call_fn_86809179?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_7_layer_call_and_return_conditional_losses_86809189?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_7_layer_call_fn_86809198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_2_layer_call_and_return_conditional_losses_86809203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_re_lu_2_layer_call_fn_86809208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_8_layer_call_and_return_conditional_losses_86809218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_8_layer_call_fn_86809227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_3_layer_call_and_return_conditional_losses_86809232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_re_lu_3_layer_call_fn_86809237?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_layer_call_and_return_conditional_losses_86809251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_reshape_layer_call_fn_86809256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_86803579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
3__inference_conv2d_transpose_layer_call_fn_86803589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
E__inference_re_lu_4_layer_call_and_return_conditional_losses_86809261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_re_lu_4_layer_call_fn_86809266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_86803623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
5__inference_conv2d_transpose_1_layer_call_fn_86803633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
E__inference_re_lu_5_layer_call_and_return_conditional_losses_86809271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_re_lu_5_layer_call_fn_86809276?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86803668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
5__inference_conv2d_transpose_2_layer_call_fn_86803678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? ?
E__inference_Decoder_layer_call_and_return_conditional_losses_86804053?
HIJKLMNOPQ4?1
*?'
!?
input_1?????????
p
? "??<
5?2
0+???????????????????????????
? ?
E__inference_Decoder_layer_call_and_return_conditional_losses_86804078?
HIJKLMNOPQ4?1
*?'
!?
input_1?????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_Decoder_layer_call_and_return_conditional_losses_86808050k
HIJKLMNOPQ.?+
$?!
?
x?????????
p
? "-?*
#? 
0?????????
? ?
E__inference_Decoder_layer_call_and_return_conditional_losses_86808138k
HIJKLMNOPQ.?+
$?!
?
x?????????
p 
? "-?*
#? 
0?????????
? ?
*__inference_Decoder_layer_call_fn_86804129v
HIJKLMNOPQ4?1
*?'
!?
input_1?????????
p
? "2?/+????????????????????????????
*__inference_Decoder_layer_call_fn_86804179v
HIJKLMNOPQ4?1
*?'
!?
input_1?????????
p 
? "2?/+????????????????????????????
*__inference_Decoder_layer_call_fn_86808163p
HIJKLMNOPQ.?+
$?!
?
x?????????
p
? "2?/+????????????????????????????
*__inference_Decoder_layer_call_fn_86808188p
HIJKLMNOPQ.?+
$?!
?
x?????????
p 
? "2?/+????????????????????????????
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803439p@ABCDEFG<?9
2?/
)?&
input_1?????????
p
? "&?#
?
0??????????
? ?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86803460p@ABCDEFG<?9
2?/
)?&
input_1?????????
p 
? "&?#
?
0??????????
? ?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86807534j@ABCDEFG6?3
,?)
#? 
x?????????
p
? "&?#
?
0??????????
? ?
N__inference_FeatureExtractor_layer_call_and_return_conditional_losses_86807568j@ABCDEFG6?3
,?)
#? 
x?????????
p 
? "&?#
?
0??????????
? ?
3__inference_FeatureExtractor_layer_call_fn_86803503c@ABCDEFG<?9
2?/
)?&
input_1?????????
p
? "????????????
3__inference_FeatureExtractor_layer_call_fn_86803545c@ABCDEFG<?9
2?/
)?&
input_1?????????
p 
? "????????????
3__inference_FeatureExtractor_layer_call_fn_86807589]@ABCDEFG6?3
,?)
#? 
x?????????
p
? "????????????
3__inference_FeatureExtractor_layer_call_fn_86807610]@ABCDEFG6?3
,?)
#? 
x?????????
p 
? "????????????
A__inference_VAE_layer_call_and_return_conditional_losses_86805439?@ABCDEFG"#()./45HIJKLMNOPQ6?3
,?)
#? 
x?????????
p
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????

?
0/4?????????
?
0/5?????????
?
0/6?????????
%?"
0/7?????????
? ?
A__inference_VAE_layer_call_and_return_conditional_losses_86805579?@ABCDEFG"#()./45HIJKLMNOPQ6?3
,?)
#? 
x?????????
p 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????

?
0/4?????????
?
0/5?????????
?
0/6?????????
%?"
0/7?????????
? ?
A__inference_VAE_layer_call_and_return_conditional_losses_86805876?@ABCDEFG"#()./45HIJKLMNOPQ<?9
2?/
)?&
input_1?????????
p
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????

?
0/4?????????
?
0/5?????????
?
0/6?????????
%?"
0/7?????????
? ?
A__inference_VAE_layer_call_and_return_conditional_losses_86806016?@ABCDEFG"#()./45HIJKLMNOPQ<?9
2?/
)?&
input_1?????????
p 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????

?
0/4?????????
?
0/5?????????
?
0/6?????????
%?"
0/7?????????
? ?
&__inference_VAE_layer_call_fn_86805658?@ABCDEFG"#()./45HIJKLMNOPQ6?3
,?)
#? 
x?????????
p
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????

?
4?????????
?
5?????????
?
6?????????
5?2
7+????????????????????????????
&__inference_VAE_layer_call_fn_86805737?@ABCDEFG"#()./45HIJKLMNOPQ6?3
,?)
#? 
x?????????
p 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????

?
4?????????
?
5?????????
?
6?????????
5?2
7+????????????????????????????
&__inference_VAE_layer_call_fn_86806095?@ABCDEFG"#()./45HIJKLMNOPQ<?9
2?/
)?&
input_1?????????
p
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????

?
4?????????
?
5?????????
?
6?????????
5?2
7+????????????????????????????
&__inference_VAE_layer_call_fn_86806174?@ABCDEFG"#()./45HIJKLMNOPQ<?9
2?/
)?&
input_1?????????
p 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????

?
4?????????
?
5?????????
?
6?????????
5?2
7+????????????????????????????
#__inference__wrapped_model_86803070?@ABCDEFG"#()./45HIJKLMNOPQ8?5
.?+
)?&
input_1?????????
? "???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
.
output_3"?
output_3?????????
.
output_4"?
output_4?????????

.
output_5"?
output_5?????????
.
output_6"?
output_6?????????
.
output_7"?
output_7?????????
6
output_8*?'
output_8??????????
__inference_call_86806313?@ABCDEFG"#()./45HIJKLMNOPQ6?3
,?)
#? 
x?????????
p
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????

?
4?????????
?
5?????????
?
6?????????
#? 
7??????????
__inference_call_86806453?@ABCDEFG"#()./45HIJKLMNOPQ6?3
,?)
#? 
x?????????
p 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????

?
4?????????
?
5?????????
?
6?????????
#? 
7??????????
__inference_call_86806715?@ABCDEFG"#()./45HIJKLMNOPQ-?*
#? 
?
x
p 
? "???
?
0
?
1
?
2
?
3

?
4
?
5
?
6
?
7?
__inference_call_86806977?@ABCDEFG"#()./45HIJKLMNOPQ-?*
#? 
?
x
p 
? "???
?
0
?
1
?
2
?
3

?
4
?
5
?
6
?
7?
__inference_call_86807239?@ABCDEFG"#()./45HIJKLMNOPQ-?*
#? 
?
x@
p 
? "???
?
0@
?
1@
?
2@
?
3@

?
4@
?
5@
?
6@
?
7@?
__inference_call_86807500?@ABCDEFG"#()./45HIJKLMNOPQ-?*
#? 
?
x@
p
? "???
?
0@
?
1@
?
2@
?
3@

?
4@
?
5@
?
6@
?
7@z
__inference_call_86807644]@ABCDEFG6?3
,?)
#? 
x?????????
p
? "???????????h
__inference_call_86807678K@ABCDEFG-?*
#? 
?
x
p 
? "?	?h
__inference_call_86807712K@ABCDEFG-?*
#? 
?
x@
p 
? "?	@?h
__inference_call_86807746K@ABCDEFG-?*
#? 
?
x@
p
? "?	@?h
__inference_call_86807780K@ABCDEFG-?*
#? 
?
x
p
? "?	?h
__inference_call_86807814K@ABCDEFG-?*
#? 
?
x
p 
? "?	?z
__inference_call_86807848]@ABCDEFG6?3
,?)
#? 
x?????????
p 
? "???????????{
__inference_call_86808276^
HIJKLMNOPQ.?+
$?!
?
x?????????
p
? " ??????????i
__inference_call_86808364L
HIJKLMNOPQ%?"
?
?
x
p 
? "?i
__inference_call_86808452L
HIJKLMNOPQ%?"
?
?
x
p 
? "?{
__inference_call_86808540^
HIJKLMNOPQ.?+
$?!
?
x?????????
p 
? " ??????????i
__inference_call_86808628L
HIJKLMNOPQ%?"
?
?
x@
p 
? "?@i
__inference_call_86808716L
HIJKLMNOPQ%?"
?
?
x@
p
? "?@?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_86809091lBC7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
+__inference_conv2d_1_layer_call_fn_86809100_BC7?4
-?*
(?%
inputs????????? 
? " ??????????@?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_86809120lDE7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
+__inference_conv2d_2_layer_call_fn_86809129_DE7?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_conv2d_layer_call_and_return_conditional_losses_86809062l@A7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_layer_call_fn_86809071_@A7?4
-?*
(?%
inputs?????????
? " ?????????? ?
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_86803623?NOI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_conv2d_transpose_1_layer_call_fn_86803633?NOI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86803668?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_2_layer_call_fn_86803678?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_86803579?LMI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_layer_call_fn_86803589?LMI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
E__inference_dense_1_layer_call_and_return_conditional_losses_86807858]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_1_layer_call_fn_86807867P0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_2_layer_call_and_return_conditional_losses_86807877]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_2_layer_call_fn_86807886P0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_3_layer_call_and_return_conditional_losses_86807896]"#0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ~
*__inference_dense_3_layer_call_fn_86807905P"#0?-
&?#
!?
inputs??????????
? "??????????
?
E__inference_dense_4_layer_call_and_return_conditional_losses_86807915]()/?,
%?"
 ?
inputs?????????

? "&?#
?
0??????????
? ~
*__inference_dense_4_layer_call_fn_86807924P()/?,
%?"
 ?
inputs?????????

? "????????????
E__inference_dense_5_layer_call_and_return_conditional_losses_86807934]./0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_5_layer_call_fn_86807943P./0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_6_layer_call_and_return_conditional_losses_86807953]450?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_6_layer_call_fn_86807962P450?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_7_layer_call_and_return_conditional_losses_86809189]HI/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_dense_7_layer_call_fn_86809198PHI/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_dense_8_layer_call_and_return_conditional_losses_86809218^JK0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_8_layer_call_fn_86809227QJK0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_layer_call_and_return_conditional_losses_86809160^FG0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_layer_call_fn_86809169QFG0?-
&?#
!?
inputs??????????
? "????????????
E__inference_flatten_layer_call_and_return_conditional_losses_86809145a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
*__inference_flatten_layer_call_fn_86809150T7?4
-?*
(?%
inputs?????????@
? "????????????
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_86809174Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_leaky_re_lu_1_layer_call_fn_86809179M0?-
&?#
!?
inputs??????????
? "????????????
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_86809134h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
.__inference_leaky_re_lu_layer_call_fn_86809139[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_re_lu_1_layer_call_and_return_conditional_losses_86809105h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_re_lu_1_layer_call_fn_86809110[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_re_lu_2_layer_call_and_return_conditional_losses_86809203Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
*__inference_re_lu_2_layer_call_fn_86809208M0?-
&?#
!?
inputs??????????
? "????????????
E__inference_re_lu_3_layer_call_and_return_conditional_losses_86809232Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
*__inference_re_lu_3_layer_call_fn_86809237M0?-
&?#
!?
inputs??????????
? "????????????
E__inference_re_lu_4_layer_call_and_return_conditional_losses_86809261?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_re_lu_4_layer_call_fn_86809266I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
E__inference_re_lu_5_layer_call_and_return_conditional_losses_86809271?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_re_lu_5_layer_call_fn_86809276I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
C__inference_re_lu_layer_call_and_return_conditional_losses_86809076h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
(__inference_re_lu_layer_call_fn_86809081[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
E__inference_reshape_layer_call_and_return_conditional_losses_86809251a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????@
? ?
*__inference_reshape_layer_call_fn_86809256T0?-
&?#
!?
inputs??????????
? " ??????????@?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803824?
HIJKLMNOPQ>?;
4?1
'?$
dense_7_input?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86803858?
HIJKLMNOPQ>?;
4?1
'?$
dense_7_input?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86808914t
HIJKLMNOPQ7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_86809002t
HIJKLMNOPQ7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
/__inference_sequential_1_layer_call_fn_86803918?
HIJKLMNOPQ>?;
4?1
'?$
dense_7_input?????????
p

 
? "2?/+????????????????????????????
/__inference_sequential_1_layer_call_fn_86803977?
HIJKLMNOPQ>?;
4?1
'?$
dense_7_input?????????
p 

 
? "2?/+????????????????????????????
/__inference_sequential_1_layer_call_fn_86809027y
HIJKLMNOPQ7?4
-?*
 ?
inputs?????????
p

 
? "2?/+????????????????????????????
/__inference_sequential_1_layer_call_fn_86809052y
HIJKLMNOPQ7?4
-?*
 ?
inputs?????????
p 

 
? "2?/+????????????????????????????
H__inference_sequential_layer_call_and_return_conditional_losses_86803245y@ABCDEFGE?B
;?8
.?+
conv2d_input?????????
p

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_86803274y@ABCDEFGE?B
;?8
.?+
conv2d_input?????????
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_86808750s@ABCDEFG??<
5?2
(?%
inputs?????????
p

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_86808784s@ABCDEFG??<
5?2
(?%
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
-__inference_sequential_layer_call_fn_86803325l@ABCDEFGE?B
;?8
.?+
conv2d_input?????????
p

 
? "????????????
-__inference_sequential_layer_call_fn_86803375l@ABCDEFGE?B
;?8
.?+
conv2d_input?????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_86808805f@ABCDEFG??<
5?2
(?%
inputs?????????
p

 
? "????????????
-__inference_sequential_layer_call_fn_86808826f@ABCDEFG??<
5?2
(?%
inputs?????????
p 

 
? "????????????
&__inference_signature_wrapper_86805178?@ABCDEFG"#()./45HIJKLMNOPQC?@
? 
9?6
4
input_1)?&
input_1?????????"???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
.
output_3"?
output_3?????????
.
output_4"?
output_4?????????

.
output_5"?
output_5?????????
.
output_6"?
output_6?????????
.
output_7"?
output_7?????????
6
output_8*?'
output_8?????????