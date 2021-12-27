use std::error::Error;
use std::io::Cursor;
use clap::{Arg, App};
use tract_onnx::{self, prelude::{SimplePlan, TypedFact, Framework, Datum, InferenceFact, InferenceModelExt, tvec, TypedOp}};

const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");
//CARGO_PKG_HOMEPAGE
const APP_NAME: &str = env!("CARGO_PKG_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");

// Image patch input:
const MODEL_INPUT_CHANNELS: usize = 3;
const MODEL_INPUT_WIDTH: usize = 256;
const MODEL_INPUT_HEIGHT: usize = 256;

// Text mask output:
const MODEL_STAGE_1_CHANNELS: usize = 1;
const MODEL_STAGE_1_WIDTH: usize = 256;
const MODEL_STAGE_1_HEIGHT: usize = 256;

fn main() {
	let matches = App::new(APP_NAME)
		.version(VERSION)
		.author(AUTHORS)
		.about(DESCRIPTION)

		// Note: Windows doesn't usually expand globstar arguments and Linux _can_ pass literal globstar.

		.arg(Arg::new("threshold")
			.short('t')
			.long("threshold")
			.takes_value(true)
			.required(false)
			.help("The minimum detection threshold for a 'text' pixel.  0 = any pixel is a text pixel.  100 = no pixel is a text pixel.  Default = 80.")
		)
		.arg(Arg::new("expr")
			.short('e')
			.long("expression")
			.takes_value(true)
			.required(false)
			.help("If specified, an expression to match inside the image files.")
		)
		//#[cfg(target_os = "windows")]
		.arg(Arg::new("expandwildcard")
			.short('g')
			.long("expandwildcard")
			//.takes_value(false))
			.help("Expand wildcards using internal GLOB support for compatibility with Windows CMD.")
		)
		.arg(Arg::new("files")
			.short('f')
			.long("files")
			.index(1)
			.multiple_occurrences(true)
			.takes_value(true)
			.required(true)
			.help("Image files to search for the given string.")
		)
		.get_matches();

	//assert!(m.is_present("verbose"));
	//assert_eq!(m.occurrences_of("verbose"), 3);
	let myfiles: Vec<String> = matches.values_of_t_or_exit("files");
	for f in myfiles {
		println!("The file passed is: {}", &f);
	}
	if let Some(thresh_str) = matches.value_of("threshold") {
		println!("Using threshold: {}", thresh_str);
	}

	//let num_str = matches.value_of("num");
	//println!("Hello, world!");

	let _model = load_detector_model();
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