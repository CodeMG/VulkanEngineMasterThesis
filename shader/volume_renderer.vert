#version 450

layout(push_constant) uniform PCs{
    float focal_length;
	float aspect_ratio;
	float n;
	float f;
	float cameraXPos;
    float cameraYPos;
    float cameraZPos;
    float cameraXAngle;
    float cameraYAngle;
    float cameraZAngle;
	int frame;
};

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

vec3 cube_centre = vec3(0,0,0);
vec3 cube_dimensions = vec3(2,2,2);
vec3 camera_position = vec3(cameraXPos,cameraYPos,cameraZPos);


layout(location = 2) out Ray vRay;
layout(location = 6) out Cube cube;


mat4 MakeProjectionMatrix(float fovy_rads, float aspect_ratio, float near, float far)
{
    float g = 1.0f / tan(fovy_rads * 0.5);
    float k = far / (far - near);

    return mat4(g / aspect_ratio,  0.0f,   0.0f,   0.0f,
                 0.0f,  g,      0.0f,   0.0f,
                 0.0f,  0.0f,   k,      -near * k,
                 0.0f,  0.0f,   1.0f,   0.0f);
}


vec2 square_positions[6] = vec2[](
    vec2(1.0, -1.0), 
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
	vec2(-1.0,-1.0),
	vec2(1.0, -1.0),
	vec2(-1.0, 1.0)
);


vec3 cube_position[36] = vec3[](
  vec3(1.0, 1.0, 1.0),
  vec3(1.0, 1.0, -1.0),
  vec3(-1.0, 1.0, -1.0),
  
  vec3(1.0, 1.0, 1.0),
  vec3(-1.0, 1.0, -1.0),
  vec3(-1.0, 1.0, 1.0),
  
  vec3(1.0, 1.0, 1.0),
  vec3(1.0, -1.0, 1.0),
  vec3(1.0, -1.0, -1.0),
  
  vec3(1.0, 1.0, 1.0),
  vec3(1.0, -1.0, -1.0),
  vec3(1.0, 1.0, -1.0),
  
  vec3(1.0, 1.0, -1.0),
  vec3(1.0, -1.0, -1.0),
  vec3(-1.0, -1.0, -1.0),
  
  vec3(1.0, 1.0, -1.0),
  vec3(-1.0, -1.0, -1.0),
  vec3(-1.0, 1.0, -1.0),
  
  vec3(-1.0, 1.0, -1.0),
  vec3(-1.0, -1.0, -1.0),
  vec3(-1.0, -1.0, 1.0),
  
  vec3(-1.0, 1.0, -1.0),
  vec3(-1.0, -1.0, 1.0),
  vec3(-1.0, 1.0, 1.0),
  
  vec3(-1.0, 1.0, 1.0),
  vec3(-1.0, -1.0, 1.0),
  vec3(1.0, -1.0, 1.0),
  
  vec3(-1.0, 1.0, 1.0),
  vec3(1.0, -1.0, 1.0),
  vec3(1.0, 1.0, 1.0),
  
  
  //culprit
  vec3(1.0, -1.0, 1.0),
  vec3(-1.0, -1.0, 1.0),
  vec3(-1.0, -1.0, -1.0),
  
  vec3(1.0, -1.0, 1.0),
  vec3(-1.0, -1.0, -1.0),
  vec3(1.0, -1.0, -1.0)
);

mat4 rot_around_X(float angle){
	return transpose(mat4(1,0,0,0,
				0,cos(angle),sin(angle),0,
				0,-sin(angle),cos(angle),0,
				0,0,0,1));
}

mat4 rot_around_Y(float angle){
	return transpose(mat4(cos(angle),0,-sin(angle),0,
				0,1,0,0,
				sin(angle),0,cos(angle),0,
				0,0,0,1));
}

mat4 rot_around_Z(float angle){
	return transpose(mat4(cos(angle),-sin(angle),0,0,
						  sin(angle),cos(angle),0,0,
						  0,0,1,0,
						  0,0,0,1));
}

void main() {
	mat4 model = transpose(mat4(1,0.0,0.0,cube_centre.x,
					  0.0,1,0.0,cube_centre.y,
					  0.0,0.0,1,cube_centre.z,
					  0.0,0.0,0.0,1.0));			  
	
	model = rot_around_X(cameraZAngle) * model;
	model = rot_around_Y(cameraYAngle)*model;
	mat4 view = transpose(mat4(1,0.0,0.0,-camera_position.x,
					  0.0,1,0.0,-camera_position.y,
					  0.0,0.0,1,-camera_position.z,
					  0.0,0.0,0.0,1.0));
					  
	
	
	vec4 cube_world_pos = model* vec4(cube_position[gl_VertexIndex],1.0);
	cube_world_pos = vec4(cube_world_pos.xyz/cube_world_pos.w,cube_world_pos.w);

	vRay.dir = normalize(cube_world_pos.xyz - camera_position);
	//vRay.dir = (rot_around_X(-cameraZAngle) * vec4(vRay.dir,1.0)).xyz;
	vRay.dir = (rot_around_Y(-cameraYAngle) * vec4(vRay.dir,1.0)).xyz;
	vRay.dir = (rot_around_X(-cameraZAngle) * vec4(vRay.dir,1.0)).xyz;
	//vRay.dir = normalize(cube_position[gl_VertexIndex]-camera_position);
    vRay.origin = camera_position;	
	vRay.vertexPos = cube_position[gl_VertexIndex];
	//vRay.vertexPos = cube_world_pos.xyz;
	
	
	cube.center = cube_centre;
	cube.dimensions = cube_dimensions;
	cube.frame = frame;			
	mat4 proj = MakeProjectionMatrix(focal_length,aspect_ratio,n,f);
    gl_Position = (transpose(proj)*(view)*(model))*vec4(cube_position[gl_VertexIndex],1.0);
}