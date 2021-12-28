use std::{cmp, f32};
use image::{GenericImageView, Pixel};
use lazy_static::lazy_static;
use std::io::Cursor;
use tract_onnx::{self, prelude::{SimplePlan, TypedFact, Tensor, tract_ndarray, Framework, Datum, InferenceFact, InferenceModelExt, tvec, TypedOp}};

// Image patch input:
const MODEL_INPUT_CHANNELS: usize = 3;
const MODEL_INPUT_WIDTH: usize = 256;
const MODEL_INPUT_HEIGHT: usize = 256;

// Text mask output:
const MODEL_STAGE_1_CHANNELS: usize = 1;
const MODEL_STAGE_1_WIDTH: usize = 256;
const MODEL_STAGE_1_HEIGHT: usize = 256;

lazy_static! {
	static ref DETECTOR_MODEL: SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>> = load_detector_model();
}

fn load_detector_model() -> SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>> {
	let model_bytes = include_bytes!("../models/stage1_mk3_256x256.onnx");
	let mut model_reader = Cursor::new(model_bytes);
	tract_onnx::onnx()
		// load the model
		.model_for_read(&mut model_reader)//.model_for_path("models/encoder_cpu.onnx")
		.expect("Failed to allocate detector model -- memory corruption?")
		// specify input type and shape
		.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, MODEL_INPUT_CHANNELS as i64, MODEL_INPUT_HEIGHT as i64, MODEL_INPUT_WIDTH as i64)))
		.expect("Failed to specify input shape.")
		// Make sure output shape is defined:
		//.with_output_fact(output_node_idx, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, MODEL_OUTPUT_CHANNELS as i64, MODEL_OUTPUT_HEIGHT as i64, MODEL_OUTPUT_WIDTH as i64)))
		//.expect("Failed to specify output shape.")
		// Quantize
		.into_optimized()
		.expect("Unable to optimize model for inference")
		// make the model runnable and fix its inputs and outputs
		.into_runnable()
		.expect("Failed make model runnable.")
}

pub fn get_image_mask<I: GenericImageView<Pixel = image::Rgb<u8>>>(img: &I, offset: (usize, usize)) -> Vec<f32> {
	let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, MODEL_INPUT_CHANNELS, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), |(_, c, y, x)| {
		let image_x = x + offset.0;
		let image_y = y + offset.1;
		if image_x < img.width() as usize && image_y < img.height() as usize {
			let pixel = img.get_pixel(image_x as u32, image_y as u32);
			let rgb = pixel.0;
			rgb[c] as f32 / 255.0
			//data.get(((x + y * MODEL_INPUT_WIDTH) * MODEL_INPUT_CHANNELS + c) as i32) as f32 / 255.0
		} else {
			0f32
		}
	}).into();

	// run the model on the input
	let result = DETECTOR_MODEL.run(tvec!(image)).unwrap();

	// Result should only have one output channel:
	let output_image: Vec<f32> = result.first().unwrap()
		.to_array_view::<f32>().unwrap()
		.iter()
		.map(|v|{ (*v).max(0f32).min(1f32) }) // Should we actually cap this?
		.collect();

	output_image
}

fn chw_to_whc<T>(input_vec: Vec<T>, width: usize, height: usize, channels: usize) -> Vec<T> where T: Copy {
	// Output image is now in CHW form.  We need to convert to WHC.
	let mut converted_output: Vec<T> = Vec::<T>::with_capacity((width*height*channels) as usize);
	//for idx in 0..output_image.len() {}
	// value(n, c, h, w) = n * CHW + c * HW + h * W + w
	// offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
	// offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
	// Convert this value index from CHW, [c*(w*h) + y*width + x] to WHC/RGB [(x+y*w)*3 + c]
	for y in 0..height {
		for x in 0..width {
			for c in 0..channels {
				// Get this position in the output_image and append it to our RGB image.
				let original_offset = (c*height*width) + (y*width) + x;
				converted_output.push(input_vec[original_offset as usize]);
			}
		}
	}

	converted_output
}

pub fn check_region_for_text<I: GenericImageView<Pixel=image::Rgb<u8>>>(img: &I, offset: (usize, usize)) -> f32 {
	let mask = get_image_mask(img, offset);

	mask.into_iter().fold(0f32, f32::max)
}

pub fn check_image_for_text<I: GenericImageView<Pixel=image::Rgb<u8>>>(img: &I, early_stop_threshold: Option<f32>) -> f32 {
	let min_threshold = if let Some(x) = early_stop_threshold {
		x
	} else {
		1.0
	};

	// Divide the image into regions and try each in parallel.
	let y_steps = img.height() as usize / MODEL_INPUT_HEIGHT;
	let x_steps = img.width() as usize / MODEL_INPUT_WIDTH;
	let mut max_response = 0.0;
	// Note there's no overlap yet.
	for y in 0..y_steps {
		for x in 0..x_steps {
			let resp = check_region_for_text(img, (x * MODEL_INPUT_WIDTH, y * MODEL_INPUT_HEIGHT));
			if resp > max_response {
				max_response = resp;
			}
			if resp >= min_threshold {
				return resp;
			}
		}
	}
	return max_response;
}

/*
Inference:
let img_width = img.get_width();
		let img_height = img.get_height();

		// image is an rgb8 but our model expects u8
		//let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);

 */