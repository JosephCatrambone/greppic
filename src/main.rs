use std::error::Error;
use std::io::Cursor;
use clap::Parser;
use tract_onnx::{self, prelude::{SimplePlan, TypedFact, Framework, Datum, InferenceFact, InferenceModelExt, tvec, TypedOp}};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
	/// Name of the person to greet
	#[clap(short, long)]
	name: String,

	/// Number of times to greet
	#[clap(short, long, default_value_t = 1)]
	count: u8,

	/// Minimum confidence for an image to be considered to 'have' text.  Min: 0, Max: 100.
	#[clap(short, long, default_value_t = 70)]
	minconf: u8,

	/// Overlap text scan
	#[clap(short, long, default_value_t = false)]
	overlap_scan: bool,
}

// Level of verbosity, default=0 (print only text)
//#[clap(short, long, parse(from_occurrences))]
//verbose: usize,

fn load_model() -> SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>> {
	let model_bytes = include_bytes!("../models/drawing_to_cat.onnx");
	let mut model_reader = Cursor::new(model_bytes);
	tract_onnx::onnx()
		// load the model
		.model_for_read(&mut model_reader)//.model_for_path("models/encoder_cpu.onnx")
		.expect("Failed to load model from models/drawing_to_animal.onnx")
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

fn main() {
	let args = Args::parse();
	println!("Hello, world!");

	let model = load_model();
}

/*
Inference:
let img_width = img.get_width();
		let img_height = img.get_height();

		// image is an rgb8 but our model expects u8
		//let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
		let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, MODEL_INPUT_CHANNELS, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), |(_, c, y, x)| {
			// Assume [r g b a r g b a r g b a ...] row major.
			//data[(x + y*MODEL_INPUT_WIDTH)*3 + c] as f32 / 255.0
			data.get(((x + y*MODEL_INPUT_WIDTH)*MODEL_INPUT_CHANNELS + c) as i32) as f32 / 255.0
		}).into();

		// run the model on the input
		let result = if let Some(mdl) = &self.model {
			Some(mdl.run(tvec!(image)).unwrap())
		} else {
			None
		};

		let output_image: Vec<u8> = result.unwrap()[0]
			.to_array_view::<f32>().unwrap()
			.iter()
			.map(|v|{ (v.max(0f32).min(1f32) * 255f32) as u8 })
			.collect();

		// Output image is now in CHW form.  We need to convert to WHC.
		let mut converted_output: Vec<u8> = Vec::<u8>::with_capacity((MODEL_OUTPUT_CHANNELS*MODEL_OUTPUT_WIDTH*MODEL_OUTPUT_HEIGHT) as usize);
		//for idx in 0..output_image.len() {}
		// value(n, c, h, w) = n * CHW + c * HW + h * W + w
		// offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
		// offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
		// Convert this value index from CHW, [c*(w*h) + y*width + x] to WHC/RGB [(x+y*w)*3 + c]
		for y in 0..MODEL_OUTPUT_HEIGHT {
			for x in 0..MODEL_OUTPUT_WIDTH {
				for c in 0..MODEL_OUTPUT_CHANNELS {
					// Get this position in the output_image and append it to our RGB image.
					let original_offset = (c*MODEL_OUTPUT_HEIGHT*MODEL_OUTPUT_WIDTH) + (y*MODEL_OUTPUT_WIDTH) + x;
					converted_output.push(output_image[original_offset as usize]);
				}
			}
		}

		let i = gdapi::Image::new();
		i.create_from_data(
			MODEL_OUTPUT_WIDTH,
			MODEL_OUTPUT_HEIGHT,
			false,
			gdapi::Image::FORMAT_RGB8,
			TypedArray::from_vec(converted_output)
		);
 */