??
??
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
?
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
?
my_model_17/conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namemy_model_17/conv2d_50/kernel
?
0my_model_17/conv2d_50/kernel/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_50/kernel*&
_output_shapes
:*
dtype0
?
my_model_17/conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namemy_model_17/conv2d_50/bias
?
.my_model_17/conv2d_50/bias/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_50/bias*
_output_shapes
:*
dtype0
?
my_model_17/conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namemy_model_17/conv2d_51/kernel
?
0my_model_17/conv2d_51/kernel/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_51/kernel*&
_output_shapes
:*
dtype0
?
my_model_17/conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namemy_model_17/conv2d_51/bias
?
.my_model_17/conv2d_51/bias/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_51/bias*
_output_shapes
:*
dtype0
?
my_model_17/conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namemy_model_17/conv2d_52/kernel
?
0my_model_17/conv2d_52/kernel/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_52/kernel*&
_output_shapes
: *
dtype0
?
my_model_17/conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namemy_model_17/conv2d_52/bias
?
.my_model_17/conv2d_52/bias/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_52/bias*
_output_shapes
: *
dtype0
?
my_model_17/conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_namemy_model_17/conv2d_53/kernel
?
0my_model_17/conv2d_53/kernel/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_53/kernel*&
_output_shapes
:  *
dtype0
?
my_model_17/conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namemy_model_17/conv2d_53/bias
?
.my_model_17/conv2d_53/bias/Read/ReadVariableOpReadVariableOpmy_model_17/conv2d_53/bias*
_output_shapes
: *
dtype0
?
my_model_17/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namemy_model_17/dense_26/kernel
?
/my_model_17/dense_26/kernel/Read/ReadVariableOpReadVariableOpmy_model_17/dense_26/kernel* 
_output_shapes
:
??*
dtype0
?
my_model_17/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namemy_model_17/dense_26/bias
?
-my_model_17/dense_26/bias/Read/ReadVariableOpReadVariableOpmy_model_17/dense_26/bias*
_output_shapes	
:?*
dtype0
?
my_model_17/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*,
shared_namemy_model_17/dense_27/kernel
?
/my_model_17/dense_27/kernel/Read/ReadVariableOpReadVariableOpmy_model_17/dense_27/kernel*
_output_shapes
:	?
*
dtype0
?
my_model_17/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namemy_model_17/dense_27/bias
?
-my_model_17/dense_27/bias/Read/ReadVariableOpReadVariableOpmy_model_17/dense_27/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
	conv1
	conv2
	max_pool1
	conv3
	conv4
	max_pool2
flatten
dropout
	d1

d2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
V
0
1
2
3
 4
!5
&6
'7
88
99
>10
?11
V
0
1
2
3
 4
!5
&6
'7
88
99
>10
?11
 
?
Dlayer_regularization_losses

Elayers
Fnon_trainable_variables
	variables
trainable_variables
regularization_losses
Glayer_metrics
Hmetrics
 
YW
VARIABLE_VALUEmy_model_17/conv2d_50/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmy_model_17/conv2d_50/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Ilayer_regularization_losses

Jlayers
Knon_trainable_variables
	variables
trainable_variables
regularization_losses
Llayer_metrics
Mmetrics
YW
VARIABLE_VALUEmy_model_17/conv2d_51/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmy_model_17/conv2d_51/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Nlayer_regularization_losses

Olayers
Pnon_trainable_variables
	variables
trainable_variables
regularization_losses
Qlayer_metrics
Rmetrics
 
 
 
?
Slayer_regularization_losses

Tlayers
Unon_trainable_variables
	variables
trainable_variables
regularization_losses
Vlayer_metrics
Wmetrics
YW
VARIABLE_VALUEmy_model_17/conv2d_52/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmy_model_17/conv2d_52/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
Xlayer_regularization_losses

Ylayers
Znon_trainable_variables
"	variables
#trainable_variables
$regularization_losses
[layer_metrics
\metrics
YW
VARIABLE_VALUEmy_model_17/conv2d_53/kernel'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmy_model_17/conv2d_53/bias%conv4/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
]layer_regularization_losses

^layers
_non_trainable_variables
(	variables
)trainable_variables
*regularization_losses
`layer_metrics
ametrics
 
 
 
?
blayer_regularization_losses

clayers
dnon_trainable_variables
,	variables
-trainable_variables
.regularization_losses
elayer_metrics
fmetrics
 
 
 
?
glayer_regularization_losses

hlayers
inon_trainable_variables
0	variables
1trainable_variables
2regularization_losses
jlayer_metrics
kmetrics
 
 
 
?
llayer_regularization_losses

mlayers
nnon_trainable_variables
4	variables
5trainable_variables
6regularization_losses
olayer_metrics
pmetrics
US
VARIABLE_VALUEmy_model_17/dense_26/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model_17/dense_26/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?
qlayer_regularization_losses

rlayers
snon_trainable_variables
:	variables
;trainable_variables
<regularization_losses
tlayer_metrics
umetrics
US
VARIABLE_VALUEmy_model_17/dense_27/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model_17/dense_27/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
?
vlayer_regularization_losses

wlayers
xnon_trainable_variables
@	variables
Atrainable_variables
Bregularization_losses
ylayer_metrics
zmetrics
 
F
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
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_17/conv2d_50/kernelmy_model_17/conv2d_50/biasmy_model_17/conv2d_51/kernelmy_model_17/conv2d_51/biasmy_model_17/conv2d_52/kernelmy_model_17/conv2d_52/biasmy_model_17/conv2d_53/kernelmy_model_17/conv2d_53/biasmy_model_17/dense_26/kernelmy_model_17/dense_26/biasmy_model_17/dense_27/kernelmy_model_17/dense_27/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_134777
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0my_model_17/conv2d_50/kernel/Read/ReadVariableOp.my_model_17/conv2d_50/bias/Read/ReadVariableOp0my_model_17/conv2d_51/kernel/Read/ReadVariableOp.my_model_17/conv2d_51/bias/Read/ReadVariableOp0my_model_17/conv2d_52/kernel/Read/ReadVariableOp.my_model_17/conv2d_52/bias/Read/ReadVariableOp0my_model_17/conv2d_53/kernel/Read/ReadVariableOp.my_model_17/conv2d_53/bias/Read/ReadVariableOp/my_model_17/dense_26/kernel/Read/ReadVariableOp-my_model_17/dense_26/bias/Read/ReadVariableOp/my_model_17/dense_27/kernel/Read/ReadVariableOp-my_model_17/dense_27/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_135323
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_17/conv2d_50/kernelmy_model_17/conv2d_50/biasmy_model_17/conv2d_51/kernelmy_model_17/conv2d_51/biasmy_model_17/conv2d_52/kernelmy_model_17/conv2d_52/biasmy_model_17/conv2d_53/kernelmy_model_17/conv2d_53/biasmy_model_17/dense_26/kernelmy_model_17/dense_26/biasmy_model_17/dense_27/kernelmy_model_17/dense_27/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_135369ݗ
?	
?
D__inference_dense_27_layer_call_and_return_conditional_losses_134594

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv2d_50_layer_call_fn_135127

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
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_1344142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_135118

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
D__inference_dense_27_layer_call_and_return_conditional_losses_135255

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_26_layer_call_and_return_conditional_losses_135236

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_135178

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_134777
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1343752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?	
?
,__inference_my_model_17_layer_call_fn_134942
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_17_layer_call_and_return_conditional_losses_1346902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
G
+__inference_flatten_14_layer_call_fn_135198

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_1345192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_27_layer_call_fn_135264

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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_1345942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_21_layer_call_fn_134387

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1343812
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_135138

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
~
)__inference_dense_26_layer_call_fn_135245

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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_1345682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_134393

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_26_layer_call_and_return_conditional_losses_134568

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_my_model_17_layer_call_fn_135078
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_17_layer_call_and_return_conditional_losses_1346902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_135215

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
G__inference_my_model_17_layer_call_and_return_conditional_losses_135049
x,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity?? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dx'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/BiasAdd~
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/Relu?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2DConv2Dconv2d_50/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/Relu?
max_pooling2d_21/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/BiasAdd~
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/Relu?
max_pooling2d_22/MaxPoolMaxPoolconv2d_53/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshape!max_pooling2d_22/MaxPool:output:0flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_14/Reshape?
dropout_3/IdentityIdentityflatten_14/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/Identity?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMuldropout_3/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/BiasAddt
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_26/Relu?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/BiasAdd?
IdentityIdentitydense_27/BiasAdd:output:0!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?

?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_134441

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_134469

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_134381

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?Q
?

!__inference__wrapped_model_134375
input_18
4my_model_17_conv2d_50_conv2d_readvariableop_resource9
5my_model_17_conv2d_50_biasadd_readvariableop_resource8
4my_model_17_conv2d_51_conv2d_readvariableop_resource9
5my_model_17_conv2d_51_biasadd_readvariableop_resource8
4my_model_17_conv2d_52_conv2d_readvariableop_resource9
5my_model_17_conv2d_52_biasadd_readvariableop_resource8
4my_model_17_conv2d_53_conv2d_readvariableop_resource9
5my_model_17_conv2d_53_biasadd_readvariableop_resource7
3my_model_17_dense_26_matmul_readvariableop_resource8
4my_model_17_dense_26_biasadd_readvariableop_resource7
3my_model_17_dense_27_matmul_readvariableop_resource8
4my_model_17_dense_27_biasadd_readvariableop_resource
identity??,my_model_17/conv2d_50/BiasAdd/ReadVariableOp?+my_model_17/conv2d_50/Conv2D/ReadVariableOp?,my_model_17/conv2d_51/BiasAdd/ReadVariableOp?+my_model_17/conv2d_51/Conv2D/ReadVariableOp?,my_model_17/conv2d_52/BiasAdd/ReadVariableOp?+my_model_17/conv2d_52/Conv2D/ReadVariableOp?,my_model_17/conv2d_53/BiasAdd/ReadVariableOp?+my_model_17/conv2d_53/Conv2D/ReadVariableOp?+my_model_17/dense_26/BiasAdd/ReadVariableOp?*my_model_17/dense_26/MatMul/ReadVariableOp?+my_model_17/dense_27/BiasAdd/ReadVariableOp?*my_model_17/dense_27/MatMul/ReadVariableOp?
+my_model_17/conv2d_50/Conv2D/ReadVariableOpReadVariableOp4my_model_17_conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+my_model_17/conv2d_50/Conv2D/ReadVariableOp?
my_model_17/conv2d_50/Conv2DConv2Dinput_13my_model_17/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
my_model_17/conv2d_50/Conv2D?
,my_model_17/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp5my_model_17_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,my_model_17/conv2d_50/BiasAdd/ReadVariableOp?
my_model_17/conv2d_50/BiasAddBiasAdd%my_model_17/conv2d_50/Conv2D:output:04my_model_17/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
my_model_17/conv2d_50/BiasAdd?
my_model_17/conv2d_50/ReluRelu&my_model_17/conv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
my_model_17/conv2d_50/Relu?
+my_model_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp4my_model_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+my_model_17/conv2d_51/Conv2D/ReadVariableOp?
my_model_17/conv2d_51/Conv2DConv2D(my_model_17/conv2d_50/Relu:activations:03my_model_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
my_model_17/conv2d_51/Conv2D?
,my_model_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp5my_model_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,my_model_17/conv2d_51/BiasAdd/ReadVariableOp?
my_model_17/conv2d_51/BiasAddBiasAdd%my_model_17/conv2d_51/Conv2D:output:04my_model_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
my_model_17/conv2d_51/BiasAdd?
my_model_17/conv2d_51/ReluRelu&my_model_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
my_model_17/conv2d_51/Relu?
$my_model_17/max_pooling2d_21/MaxPoolMaxPool(my_model_17/conv2d_51/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2&
$my_model_17/max_pooling2d_21/MaxPool?
+my_model_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp4my_model_17_conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+my_model_17/conv2d_52/Conv2D/ReadVariableOp?
my_model_17/conv2d_52/Conv2DConv2D-my_model_17/max_pooling2d_21/MaxPool:output:03my_model_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
my_model_17/conv2d_52/Conv2D?
,my_model_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp5my_model_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,my_model_17/conv2d_52/BiasAdd/ReadVariableOp?
my_model_17/conv2d_52/BiasAddBiasAdd%my_model_17/conv2d_52/Conv2D:output:04my_model_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
my_model_17/conv2d_52/BiasAdd?
my_model_17/conv2d_52/ReluRelu&my_model_17/conv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
my_model_17/conv2d_52/Relu?
+my_model_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp4my_model_17_conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+my_model_17/conv2d_53/Conv2D/ReadVariableOp?
my_model_17/conv2d_53/Conv2DConv2D(my_model_17/conv2d_52/Relu:activations:03my_model_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
my_model_17/conv2d_53/Conv2D?
,my_model_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp5my_model_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,my_model_17/conv2d_53/BiasAdd/ReadVariableOp?
my_model_17/conv2d_53/BiasAddBiasAdd%my_model_17/conv2d_53/Conv2D:output:04my_model_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
my_model_17/conv2d_53/BiasAdd?
my_model_17/conv2d_53/ReluRelu&my_model_17/conv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
my_model_17/conv2d_53/Relu?
$my_model_17/max_pooling2d_22/MaxPoolMaxPool(my_model_17/conv2d_53/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2&
$my_model_17/max_pooling2d_22/MaxPool?
my_model_17/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
my_model_17/flatten_14/Const?
my_model_17/flatten_14/ReshapeReshape-my_model_17/max_pooling2d_22/MaxPool:output:0%my_model_17/flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2 
my_model_17/flatten_14/Reshape?
my_model_17/dropout_3/IdentityIdentity'my_model_17/flatten_14/Reshape:output:0*
T0*(
_output_shapes
:??????????2 
my_model_17/dropout_3/Identity?
*my_model_17/dense_26/MatMul/ReadVariableOpReadVariableOp3my_model_17_dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*my_model_17/dense_26/MatMul/ReadVariableOp?
my_model_17/dense_26/MatMulMatMul'my_model_17/dropout_3/Identity:output:02my_model_17/dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
my_model_17/dense_26/MatMul?
+my_model_17/dense_26/BiasAdd/ReadVariableOpReadVariableOp4my_model_17_dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+my_model_17/dense_26/BiasAdd/ReadVariableOp?
my_model_17/dense_26/BiasAddBiasAdd%my_model_17/dense_26/MatMul:product:03my_model_17/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
my_model_17/dense_26/BiasAdd?
my_model_17/dense_26/ReluRelu%my_model_17/dense_26/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
my_model_17/dense_26/Relu?
*my_model_17/dense_27/MatMul/ReadVariableOpReadVariableOp3my_model_17_dense_27_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*my_model_17/dense_27/MatMul/ReadVariableOp?
my_model_17/dense_27/MatMulMatMul'my_model_17/dense_26/Relu:activations:02my_model_17/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
my_model_17/dense_27/MatMul?
+my_model_17/dense_27/BiasAdd/ReadVariableOpReadVariableOp4my_model_17_dense_27_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+my_model_17/dense_27/BiasAdd/ReadVariableOp?
my_model_17/dense_27/BiasAddBiasAdd%my_model_17/dense_27/MatMul:product:03my_model_17/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
my_model_17/dense_27/BiasAdd?
IdentityIdentity%my_model_17/dense_27/BiasAdd:output:0-^my_model_17/conv2d_50/BiasAdd/ReadVariableOp,^my_model_17/conv2d_50/Conv2D/ReadVariableOp-^my_model_17/conv2d_51/BiasAdd/ReadVariableOp,^my_model_17/conv2d_51/Conv2D/ReadVariableOp-^my_model_17/conv2d_52/BiasAdd/ReadVariableOp,^my_model_17/conv2d_52/Conv2D/ReadVariableOp-^my_model_17/conv2d_53/BiasAdd/ReadVariableOp,^my_model_17/conv2d_53/Conv2D/ReadVariableOp,^my_model_17/dense_26/BiasAdd/ReadVariableOp+^my_model_17/dense_26/MatMul/ReadVariableOp,^my_model_17/dense_27/BiasAdd/ReadVariableOp+^my_model_17/dense_27/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2\
,my_model_17/conv2d_50/BiasAdd/ReadVariableOp,my_model_17/conv2d_50/BiasAdd/ReadVariableOp2Z
+my_model_17/conv2d_50/Conv2D/ReadVariableOp+my_model_17/conv2d_50/Conv2D/ReadVariableOp2\
,my_model_17/conv2d_51/BiasAdd/ReadVariableOp,my_model_17/conv2d_51/BiasAdd/ReadVariableOp2Z
+my_model_17/conv2d_51/Conv2D/ReadVariableOp+my_model_17/conv2d_51/Conv2D/ReadVariableOp2\
,my_model_17/conv2d_52/BiasAdd/ReadVariableOp,my_model_17/conv2d_52/BiasAdd/ReadVariableOp2Z
+my_model_17/conv2d_52/Conv2D/ReadVariableOp+my_model_17/conv2d_52/Conv2D/ReadVariableOp2\
,my_model_17/conv2d_53/BiasAdd/ReadVariableOp,my_model_17/conv2d_53/BiasAdd/ReadVariableOp2Z
+my_model_17/conv2d_53/Conv2D/ReadVariableOp+my_model_17/conv2d_53/Conv2D/ReadVariableOp2Z
+my_model_17/dense_26/BiasAdd/ReadVariableOp+my_model_17/dense_26/BiasAdd/ReadVariableOp2X
*my_model_17/dense_26/MatMul/ReadVariableOp*my_model_17/dense_26/MatMul/ReadVariableOp2Z
+my_model_17/dense_27/BiasAdd/ReadVariableOp+my_model_17/dense_27/BiasAdd/ReadVariableOp2X
*my_model_17/dense_27/MatMul/ReadVariableOp*my_model_17/dense_27/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?	
?
,__inference_my_model_17_layer_call_fn_134913
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_17_layer_call_and_return_conditional_losses_1346902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
c
*__inference_dropout_3_layer_call_fn_135220

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1345392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_22_layer_call_fn_134399

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1343932
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_135210

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_135158

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_134544

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_my_model_17_layer_call_fn_135107
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_17_layer_call_and_return_conditional_losses_1346902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?-
?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134690
x
conv2d_50_134655
conv2d_50_134657
conv2d_51_134660
conv2d_51_134662
conv2d_52_134666
conv2d_52_134668
conv2d_53_134671
conv2d_53_134673
dense_26_134679
dense_26_134681
dense_27_134684
dense_27_134686
identity??!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCallxconv2d_50_134655conv2d_50_134657*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_1344142#
!conv2d_50/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0conv2d_51_134660conv2d_51_134662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_1344412#
!conv2d_51/StatefulPartitionedCall?
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1343812"
 max_pooling2d_21/PartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_52_134666conv2d_52_134668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_1344692#
!conv2d_52/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_134671conv2d_53_134673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_1344962#
!conv2d_53/StatefulPartitionedCall?
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1343932"
 max_pooling2d_22/PartitionedCall?
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_1345192
flatten_14/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1345442
dropout_3/PartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_26_134679dense_26_134681*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_1345682"
 dense_26/StatefulPartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_134684dense_27_134686*
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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_1345942"
 dense_27/StatefulPartitionedCall?
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?K
?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134834
input_1,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity?? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dinput_1'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/BiasAdd~
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/Relu?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2DConv2Dconv2d_50/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/Relu?
max_pooling2d_21/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/BiasAdd~
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/Relu?
max_pooling2d_22/MaxPoolMaxPoolconv2d_53/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshape!max_pooling2d_22/MaxPool:output:0flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_14/Reshapew
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMulflatten_14/Reshape:output:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul}
dropout_3/dropout/ShapeShapeflatten_14/Reshape:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul_1?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/BiasAddt
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_26/Relu?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/BiasAdd?
IdentityIdentitydense_27/BiasAdd:output:0!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?

*__inference_conv2d_52_layer_call_fn_135167

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
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_1344692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_134539

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
__inference__traced_save_135323
file_prefix;
7savev2_my_model_17_conv2d_50_kernel_read_readvariableop9
5savev2_my_model_17_conv2d_50_bias_read_readvariableop;
7savev2_my_model_17_conv2d_51_kernel_read_readvariableop9
5savev2_my_model_17_conv2d_51_bias_read_readvariableop;
7savev2_my_model_17_conv2d_52_kernel_read_readvariableop9
5savev2_my_model_17_conv2d_52_bias_read_readvariableop;
7savev2_my_model_17_conv2d_53_kernel_read_readvariableop9
5savev2_my_model_17_conv2d_53_bias_read_readvariableop:
6savev2_my_model_17_dense_26_kernel_read_readvariableop8
4savev2_my_model_17_dense_26_bias_read_readvariableop:
6savev2_my_model_17_dense_27_kernel_read_readvariableop8
4savev2_my_model_17_dense_27_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_my_model_17_conv2d_50_kernel_read_readvariableop5savev2_my_model_17_conv2d_50_bias_read_readvariableop7savev2_my_model_17_conv2d_51_kernel_read_readvariableop5savev2_my_model_17_conv2d_51_bias_read_readvariableop7savev2_my_model_17_conv2d_52_kernel_read_readvariableop5savev2_my_model_17_conv2d_52_bias_read_readvariableop7savev2_my_model_17_conv2d_53_kernel_read_readvariableop5savev2_my_model_17_conv2d_53_bias_read_readvariableop6savev2_my_model_17_dense_26_kernel_read_readvariableop4savev2_my_model_17_dense_26_bias_read_readvariableop6savev2_my_model_17_dense_27_kernel_read_readvariableop4savev2_my_model_17_dense_27_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : :  : :
??:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?
F
*__inference_dropout_3_layer_call_fn_135225

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1345442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?6
?
"__inference__traced_restore_135369
file_prefix1
-assignvariableop_my_model_17_conv2d_50_kernel1
-assignvariableop_1_my_model_17_conv2d_50_bias3
/assignvariableop_2_my_model_17_conv2d_51_kernel1
-assignvariableop_3_my_model_17_conv2d_51_bias3
/assignvariableop_4_my_model_17_conv2d_52_kernel1
-assignvariableop_5_my_model_17_conv2d_52_bias3
/assignvariableop_6_my_model_17_conv2d_53_kernel1
-assignvariableop_7_my_model_17_conv2d_53_bias2
.assignvariableop_8_my_model_17_dense_26_kernel0
,assignvariableop_9_my_model_17_dense_26_bias3
/assignvariableop_10_my_model_17_dense_27_kernel1
-assignvariableop_11_my_model_17_dense_27_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp-assignvariableop_my_model_17_conv2d_50_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_my_model_17_conv2d_50_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_my_model_17_conv2d_51_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_my_model_17_conv2d_51_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_my_model_17_conv2d_52_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_my_model_17_conv2d_52_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_my_model_17_conv2d_53_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_my_model_17_conv2d_53_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_my_model_17_dense_26_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_my_model_17_dense_26_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_my_model_17_dense_27_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_my_model_17_dense_27_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
?

*__inference_conv2d_51_layer_call_fn_135147

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
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_1344412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
b
F__inference_flatten_14_layer_call_and_return_conditional_losses_134519

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?K
?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134999
x,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity?? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dx'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/BiasAdd~
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/Relu?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2DConv2Dconv2d_50/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/Relu?
max_pooling2d_21/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/BiasAdd~
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/Relu?
max_pooling2d_22/MaxPoolMaxPoolconv2d_53/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshape!max_pooling2d_22/MaxPool:output:0flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_14/Reshapew
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMulflatten_14/Reshape:output:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul}
dropout_3/dropout/ShapeShapeflatten_14/Reshape:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul_1?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/BiasAddt
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_26/Relu?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/BiasAdd?
IdentityIdentitydense_27/BiasAdd:output:0!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?

?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_134414

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_53_layer_call_fn_135187

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
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_1344962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_134496

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
F__inference_flatten_14_layer_call_and_return_conditional_losses_135193

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?A
?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134884
input_1,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity?? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dinput_1'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/BiasAdd~
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_50/Relu?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2DConv2Dconv2d_50/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_51/Relu?
max_pooling2d_21/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_52/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/BiasAdd~
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_53/Relu?
max_pooling2d_22/MaxPoolMaxPoolconv2d_53/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshape!max_pooling2d_22/MaxPool:output:0flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_14/Reshape?
dropout_3/IdentityIdentityflatten_14/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/Identity?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMuldropout_3/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_26/BiasAddt
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_26/Relu?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_27/BiasAdd?
IdentityIdentitydense_27/BiasAdd:output:0!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????  <
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
	conv1
	conv2
	max_pool1
	conv3
	conv4
	max_pool2
flatten
dropout
	d1

d2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
{_default_save_signature
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "MyModel", "name": "my_model_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
~__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [254, 32, 32, 3]}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [254, 32, 32, 16]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [254, 16, 16, 16]}}
?	

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [254, 16, 16, 32]}}
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
?

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [254, 2048]}}
?

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [254, 512]}}
v
0
1
2
3
 4
!5
&6
'7
88
99
>10
?11"
trackable_list_wrapper
v
0
1
2
3
 4
!5
&6
'7
88
99
>10
?11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_regularization_losses

Elayers
Fnon_trainable_variables
	variables
trainable_variables
regularization_losses
Glayer_metrics
Hmetrics
|__call__
{_default_save_signature
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
6:42my_model_17/conv2d_50/kernel
(:&2my_model_17/conv2d_50/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ilayer_regularization_losses

Jlayers
Knon_trainable_variables
	variables
trainable_variables
regularization_losses
Llayer_metrics
Mmetrics
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
6:42my_model_17/conv2d_51/kernel
(:&2my_model_17/conv2d_51/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nlayer_regularization_losses

Olayers
Pnon_trainable_variables
	variables
trainable_variables
regularization_losses
Qlayer_metrics
Rmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Slayer_regularization_losses

Tlayers
Unon_trainable_variables
	variables
trainable_variables
regularization_losses
Vlayer_metrics
Wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4 2my_model_17/conv2d_52/kernel
(:& 2my_model_17/conv2d_52/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xlayer_regularization_losses

Ylayers
Znon_trainable_variables
"	variables
#trainable_variables
$regularization_losses
[layer_metrics
\metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4  2my_model_17/conv2d_53/kernel
(:& 2my_model_17/conv2d_53/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
]layer_regularization_losses

^layers
_non_trainable_variables
(	variables
)trainable_variables
*regularization_losses
`layer_metrics
ametrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
blayer_regularization_losses

clayers
dnon_trainable_variables
,	variables
-trainable_variables
.regularization_losses
elayer_metrics
fmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
glayer_regularization_losses

hlayers
inon_trainable_variables
0	variables
1trainable_variables
2regularization_losses
jlayer_metrics
kmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
llayer_regularization_losses

mlayers
nnon_trainable_variables
4	variables
5trainable_variables
6regularization_losses
olayer_metrics
pmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-
??2my_model_17/dense_26/kernel
(:&?2my_model_17/dense_26/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
qlayer_regularization_losses

rlayers
snon_trainable_variables
:	variables
;trainable_variables
<regularization_losses
tlayer_metrics
umetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,	?
2my_model_17/dense_27/kernel
':%
2my_model_17/dense_27/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vlayer_regularization_losses

wlayers
xnon_trainable_variables
@	variables
Atrainable_variables
Bregularization_losses
ylayer_metrics
zmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
f
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
9"
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
?2?
!__inference__wrapped_model_134375?
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
input_1?????????  
?2?
,__inference_my_model_17_layer_call_fn_134942
,__inference_my_model_17_layer_call_fn_135078
,__inference_my_model_17_layer_call_fn_134913
,__inference_my_model_17_layer_call_fn_135107?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134884
G__inference_my_model_17_layer_call_and_return_conditional_losses_134834
G__inference_my_model_17_layer_call_and_return_conditional_losses_134999
G__inference_my_model_17_layer_call_and_return_conditional_losses_135049?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_50_layer_call_fn_135127?
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
E__inference_conv2d_50_layer_call_and_return_conditional_losses_135118?
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
*__inference_conv2d_51_layer_call_fn_135147?
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
E__inference_conv2d_51_layer_call_and_return_conditional_losses_135138?
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
1__inference_max_pooling2d_21_layer_call_fn_134387?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_134381?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_52_layer_call_fn_135167?
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
E__inference_conv2d_52_layer_call_and_return_conditional_losses_135158?
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
*__inference_conv2d_53_layer_call_fn_135187?
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
E__inference_conv2d_53_layer_call_and_return_conditional_losses_135178?
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
1__inference_max_pooling2d_22_layer_call_fn_134399?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_134393?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_flatten_14_layer_call_fn_135198?
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
F__inference_flatten_14_layer_call_and_return_conditional_losses_135193?
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
*__inference_dropout_3_layer_call_fn_135220
*__inference_dropout_3_layer_call_fn_135225?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_3_layer_call_and_return_conditional_losses_135210
E__inference_dropout_3_layer_call_and_return_conditional_losses_135215?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_26_layer_call_fn_135245?
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
D__inference_dense_26_layer_call_and_return_conditional_losses_135236?
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
)__inference_dense_27_layer_call_fn_135264?
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
D__inference_dense_27_layer_call_and_return_conditional_losses_135255?
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
?B?
$__inference_signature_wrapper_134777input_1"?
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
 ?
!__inference__wrapped_model_134375} !&'89>?8?5
.?+
)?&
input_1?????????  
? "3?0
.
output_1"?
output_1?????????
?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_135118l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_50_layer_call_fn_135127_7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_135138l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_51_layer_call_fn_135147_7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_135158l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_52_layer_call_fn_135167_ !7?4
-?*
(?%
inputs?????????
? " ?????????? ?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_135178l&'7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_53_layer_call_fn_135187_&'7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_dense_26_layer_call_and_return_conditional_losses_135236^890?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_26_layer_call_fn_135245Q890?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_27_layer_call_and_return_conditional_losses_135255]>?0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? }
)__inference_dense_27_layer_call_fn_135264P>?0?-
&?#
!?
inputs??????????
? "??????????
?
E__inference_dropout_3_layer_call_and_return_conditional_losses_135210^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
E__inference_dropout_3_layer_call_and_return_conditional_losses_135215^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? 
*__inference_dropout_3_layer_call_fn_135220Q4?1
*?'
!?
inputs??????????
p
? "???????????
*__inference_dropout_3_layer_call_fn_135225Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_flatten_14_layer_call_and_return_conditional_losses_135193a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
+__inference_flatten_14_layer_call_fn_135198T7?4
-?*
(?%
inputs????????? 
? "????????????
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_134381?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_21_layer_call_fn_134387?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_134393?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_22_layer_call_fn_134399?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_my_model_17_layer_call_and_return_conditional_losses_134834s !&'89>?<?9
2?/
)?&
input_1?????????  
p
? "%?"
?
0?????????

? ?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134884s !&'89>?<?9
2?/
)?&
input_1?????????  
p 
? "%?"
?
0?????????

? ?
G__inference_my_model_17_layer_call_and_return_conditional_losses_134999m !&'89>?6?3
,?)
#? 
x?????????  
p
? "%?"
?
0?????????

? ?
G__inference_my_model_17_layer_call_and_return_conditional_losses_135049m !&'89>?6?3
,?)
#? 
x?????????  
p 
? "%?"
?
0?????????

? ?
,__inference_my_model_17_layer_call_fn_134913f !&'89>?<?9
2?/
)?&
input_1?????????  
p
? "??????????
?
,__inference_my_model_17_layer_call_fn_134942f !&'89>?<?9
2?/
)?&
input_1?????????  
p 
? "??????????
?
,__inference_my_model_17_layer_call_fn_135078` !&'89>?6?3
,?)
#? 
x?????????  
p
? "??????????
?
,__inference_my_model_17_layer_call_fn_135107` !&'89>?6?3
,?)
#? 
x?????????  
p 
? "??????????
?
$__inference_signature_wrapper_134777? !&'89>?C?@
? 
9?6
4
input_1)?&
input_1?????????  "3?0
.
output_1"?
output_1?????????
