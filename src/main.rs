mod detect;

use std::error::Error;
use std::io::{self, BufWriter, Write};
use clap::{Arg, App};
use detect::check_region_for_text;
use crate::detect::check_image_for_text;


const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");
//CARGO_PKG_HOMEPAGE
const APP_NAME: &str = env!("CARGO_PKG_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");


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

	// Get a STDOUT channel for writing results.
	let stdout = io::stdout();
	let mut console_out = stdout.lock();
	let mut buffered_out = BufWriter::new(console_out);

	//
	let thresh = matches.value_of("threshold").unwrap_or("70").parse::<u8>().expect("Failed to parse value for threshold.  Please select a value in the rage of 0 to 100.") as f32 / 100f32;

	//assert!(m.is_present("verbose")); assert_eq!(m.occurrences_of("verbose"), 3);
	let files: Vec<String> = matches.values_of_t_or_exit("files");
	for f in files {
		//println!("The file passed is: {}", &f);
		if let Ok(img) = image::open(&f) {
			let conf = check_image_for_text(&img.to_rgb8(), Some(thresh));
			if conf > thresh {
				buffered_out.write((format!("{} contains text.  Confidence: {}%", &f, conf)).as_ref());
			}
		}
	}

	//let num_str = matches.value_of("num");
	//println!("Hello, world!");
}
