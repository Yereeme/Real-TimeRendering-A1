#version 450

layout(location = 0) in vec2 position;

layout(location = 0) out vec4 outColor; //specify color output to the fragment shader, location 0 is first color output of render pass

const float PI = 3.14159265359; //just pi for math

// vec3 hsv2rgb(vec3 c) {
//     vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
//     vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
//     return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
// 	}

layout(push_constant) uniform Push{
	float time; //let fragment shader accept  PUSH CONSTANT
};

void main() {
	
	// vec2 grid = vec2(12.0, 10.0); //collumn and rows
	vec2 uv = position;
	// vec2 cell = floor(uv * grid);
	// float hash = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
	// float rot = (hash - 0.5) * 0.8; //random rotation per cell
	// float size = mix(1.0, 2.0, hash); //random size per cell

	// uv.x += 0.5 / grid.x * mod(cell.y, 2.0) ;

	//local coodinates inside the cell (0...1)
	// vec2 local = fract(uv * grid); //tiling, fract keep decimal, multiplying make uv 0.6 and 0.4
	
	//center to -2..2 to draw
	// vec2 p = (local * 4.0 - 2.0) * size;
	//p.x *= res.x / res.y; aspect fix

	// float r = length(p); //distance from center p the further r from p the higher the value
	// float a = atan(p.y, p.x); //angle around center, give direction (-pi-pi)

	// a += rot + time; //rotate star/ animate 

	// star
	// float k = 5.0; //number of points
	// float t = cos(a * k) * 0.4 + 1.6; //make repeating wave as i go around the circle, get 5 around circle
	// float star = smoothstep(t, t - 0.02, r); //check if pixel in star or not

	//background gradient
	float g = uv.y; //verticaql pos
	vec3 bgBottom = vec3(0.4, 0.6, 1.0); //bottom color
	vec3 bgTop = vec3(0.0, 0.0, 0.0); //top color
	vec3 bg = mix(bgBottom, bgTop, g); //full background linear blend

	// float hue = fract(uv.x + uv.y) + time; //accross screen diagonal
	// vec3 starColor = hsv2rgb(vec3(hue, 0.8,1.0)); //hue saturation value

	// float twinkle = 0.75 + 0.25 * sin(time * 3.0 + hash * 6.2832); //
	// starColor *= twinkle;

	// vec3 color = mix(bg, starColor, star);
	// outColor = vec4(color, 1.0);

	outColor = vec4(bg, 1.0);
}
