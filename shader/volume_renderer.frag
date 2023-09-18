#version 450

layout(location = 0) out vec4 outColor;
layout(origin_upper_left) in vec4 gl_FragCoord;

layout(set = 1, binding = 1) uniform sampler3D texSampler;

//layout(set=2, binding = 2) uniform TF{
//	vec4[8] tfValues;
//} tf;

layout(set=2, binding = 2) uniform sampler1D tfSampler;

struct Ray {
  vec3 origin;
  vec3 dir;
  vec3 vertexPos;
};

struct Cube{
	vec3 center;
	vec3 dimensions;
	int frame;
};

layout(location = 2) in Ray vRay;
layout(location = 6) flat in Cube cube;



bool pointInCube(vec3 point){

	vec3 topLeftEdge = 0 - 0.5*(cube.dimensions);
	vec3 bottomRightEdge = 0 + 0.5*(cube.dimensions);
	if(point.x >= topLeftEdge.x && point.y >= topLeftEdge.y && point.z >= topLeftEdge.z && point.x <= bottomRightEdge.x && point.y <= bottomRightEdge.y && point.z <= bottomRightEdge.z){
		return true;
	}
	return false;
}

vec4 transfer_function(float f){
    //float alpha = pow(f-0.5,4)*20;
	//vec4 col = vec4(f,f,f,alpha);
	//float a = tf.tfValues[0][0];
	//int selectedBin = int(f * 8);
	//vec4 col = vec4(tf.tfValues[0][0],tf.tfValues[0][1],tf.tfValues[0][2],tf.tfValues[0][3]);
	//vec4 col = vec4(tf.tfValues[selectedBin].r,tf.tfValues[selectedBin].g,tf.tfValues[selectedBin].b,tf.tfValues[selectedBin].a);
	vec4 col = vec4(0.0,0.0,0.0,0.0);
	col = texture(tfSampler,f+0.05); //The +0.05 solves a bug with the sampler
	return col;
}

vec4 backwards_sample(vec3 start, vec3 dir){
	vec4 col = vec4(0.0,0.0,0.0,0.0);
	float step_delta = 0.02;
	vec3 last_sample_point = start;
	int count = 0;
	
	//First find the end
	while(pointInCube(last_sample_point)){
		count += 1;
		last_sample_point = last_sample_point + (dir * step_delta);
	}
	last_sample_point = last_sample_point - (dir * step_delta);
	//Now backwards sample
	vec3 current = last_sample_point;
	float alpha = 0;
	for(int i = 0; i < count;i++){
		vec4 new_col = vec4(0.0,0.0,0.0,0.0);
		float t = texture(texSampler,(current+vec3(1.0,1.0,1.0))/2).x;
		vec4 c = transfer_function(t);
		alpha = c.w;
		//new_col.xyz = col.xyz + (1 - col.w)*c.xyz*t;
		new_col.w = alpha + ((1 - alpha)*col.w);
		new_col.xyz = (c.rgb * alpha + col.rgb * col.w * (1 - alpha)) / (new_col.w+0.000001);
		
		
		
		col.xyz = new_col.xyz;
		col.w = new_col.w;
		current =  current - (dir*step_delta);
	}
	//return vec4(count/255.0,count/255.0,count/255.0,1.0); 
	//return vec4(current.x,current.y,current.z,1.0);
	//col.w = 1.0;
	return col;
	//return vec4(col.w,col.w,col.w,1.0);
}

void main() {
	vec2 pos = vec2(gl_FragCoord.x,gl_FragCoord.y);
	vec4 col = vec4(1.0,0.0,0.0,1.0); //black default color
	//col = backwards_sample(vec2(gl_FragCoord.x,gl_FragCoord.y));
	//col = vec4(gl_FragCoord.x/100,gl_FragCoord.y/100,0.0,1.0);

	vec3 cameraPosition = vec3(0.0,0.0,0.0);
	
	
	
	col = backwards_sample(vRay.vertexPos,vRay.dir);
	//col = vec4(vRay.vertexPos,1.0);
	//col = vec4((vRay.dir+vec3(1))/2,1.0);
	//col = vec4(normalize((vRay.dir+vec3(1))/2),1.0);
	//col = sample_slice(vec3(pos.x,pos.y,10));
    outColor = col;
}