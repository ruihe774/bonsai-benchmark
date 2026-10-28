#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(buffer_reference,scalar,buffer_reference_align=4) buffer F { float d[]; };
layout(buffer_reference,scalar,buffer_reference_align=1) readonly buffer B { uint8_t d[]; };
layout(buffer_reference,scalar,buffer_reference_align=2) buffer H { float16_t d[]; };
layout(buffer_reference,scalar,buffer_reference_align=4) buffer I { uint d[]; };
layout(push_constant,scalar) uniform Params {
 uint64_t w,x,y,z,s,t,u,v;
 uint a,b,c,d,e,f,g,h;
} p;
float sigmoid(float x) { return 1.0/(1.0+exp(-x)); }
float silu(float x) { return x*sigmoid(x); }
