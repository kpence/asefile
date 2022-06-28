use crate::cel::CelId;
use crate::external_file::{ExternalFile, ExternalFilesById};
use crate::layer::{LayerData, LayersData};
use crate::pixel::{Pixels, RawPixels};
use crate::reader::AseReader;
use crate::slice::Slice;
use crate::tileset::{Tileset, TilesetsById};
use crate::user_data::UserData;
use crate::{error::AsepriteParseError, AsepriteFile, PixelFormat};
use crate::bytes_builder::BytesBuilder;
use crate::tags::AnimationDirection;
use log::debug;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::collections::{HashMap, BinaryHeap};
use flate2::read::ZlibEncoder;

use crate::Result;
use crate::{cel, color_profile, layer, palette, slice, tags, user_data, Tag};

struct ParseInfo {
    palette: Option<Arc<palette::ColorPalette>>,
    color_profile: Option<color_profile::ColorProfile>,
    layers: Vec<LayerData>,
    framedata: cel::CelsData<RawPixels>, // Vec<Vec<cel::RawCel>>,
    frame_times: Vec<u16>,
    old_frame_id_to_be_deleted: Vec<u16>,
    tags: Option<Vec<Tag>>,
    external_files: ExternalFilesById,
    tilesets: TilesetsById<RawPixels>,
    sprite_user_data: Option<UserData>,
    user_data_context: Option<UserDataContext>,
    slices: Vec<Slice>,
}

impl ParseInfo {
    fn new(num_frames: u16, default_frame_time: u16) -> Self {
        Self {
            palette: None,
            color_profile: None,
            layers: Vec::new(),
            framedata: cel::CelsData::new(num_frames as u32),
            frame_times: vec![default_frame_time; num_frames as usize],
            old_frame_id_to_be_deleted: vec![],
            tags: None,
            external_files: ExternalFilesById::new(),
            tilesets: TilesetsById::new(),
            sprite_user_data: None,
            user_data_context: None,
            slices: Vec::new(),
        }
    }

    fn add_cel(&mut self, frame_id: u16, cel: cel::RawCel<RawPixels>) -> Result<()> {
        let cel_id = CelId {
            frame: frame_id,
            layer: cel.data.layer_index,
        };
        self.framedata.add_cel(frame_id, cel)?;
        self.user_data_context = Some(UserDataContext::CelId(cel_id));
        Ok(())
    }

    fn add_layer(&mut self, layer_data: LayerData) {
        let idx = self.layers.len();
        self.layers.push(layer_data);
        self.user_data_context = Some(UserDataContext::LayerIndex(idx as u32));
    }

    fn add_tags(&mut self, tags: Vec<Tag>) {
        self.tags = Some(tags);
        self.user_data_context = Some(UserDataContext::TagIndex(0));
    }

    fn add_external_files(&mut self, files: Vec<ExternalFile>) {
        for external_file in files {
            self.external_files.add(external_file);
        }
    }

    fn set_tag_user_data(&mut self, user_data: UserData, tag_index: u16) -> Result<()> {
        let tags = self.tags.as_mut().ok_or_else(|| {
            AsepriteParseError::InternalError(
                "No tags data found when resolving Tags chunk context".into(),
            )
        })?;
        let tag = tags.get_mut(tag_index as usize).ok_or_else(|| {
            AsepriteParseError::InternalError(format!(
                "Invalid tag index stored in chunk context: {}",
                tag_index
            ))
        })?;
        tag.set_user_data(user_data);
        self.user_data_context = Some(UserDataContext::TagIndex(tag_index + 1));
        Ok(())
    }

    fn add_user_data(&mut self, user_data: UserData) -> Result<()> {
        let user_data_context = self.user_data_context.ok_or_else(|| {
            AsepriteParseError::InvalidInput(
                "Found dangling user data chunk. Expected a previous chunk to attach user data"
                    .into(),
            )
        })?;
        match user_data_context {
            UserDataContext::CelId(cel_id) => {
                let cel = self.framedata.cel_mut(&cel_id).ok_or_else(|| {
                    AsepriteParseError::InternalError(format!(
                        "Invalid cel id stored in chunk context: {}",
                        cel_id
                    ))
                })?;
                cel.user_data = Some(user_data);
            }
            UserDataContext::LayerIndex(layer_index) => {
                let layer = self.layers.get_mut(layer_index as usize).ok_or_else(|| {
                    AsepriteParseError::InternalError(format!(
                        "Invalid layer id stored in chunk context: {}",
                        layer_index
                    ))
                })?;
                layer.user_data = Some(user_data);
            }
            UserDataContext::OldPalette => {
                self.sprite_user_data = Some(user_data);
            }
            UserDataContext::TagIndex(tag_index) => {
                self.set_tag_user_data(user_data, tag_index)?;
            }
            UserDataContext::SliceIndex(slice_idx) => {
                let slice = self.slices.get_mut(slice_idx as usize).ok_or_else(|| {
                    AsepriteParseError::InternalError(format!(
                        "Invalid slice index stored in chunk context: {}",
                        slice_idx
                    ))
                })?;
                slice.user_data = Some(user_data);
            }
        }
        Ok(())
    }

    fn add_slice(&mut self, slice: Slice) {
        let context_idx = self.slices.len();
        self.slices.push(slice);
        self.user_data_context = Some(UserDataContext::SliceIndex(context_idx as u32));
    }

    // Validate moves the ParseInfo data into an intermediate ValidatedParseInfo struct,
    // which is then used to create the AsepriteFile.
    fn validate(self, pixel_format: &PixelFormat) -> Result<ValidatedParseInfo> {
        let layers = LayersData::from_vec(self.layers)?;

        let tilesets = self.tilesets;
        let palette = self.palette;
        let tilesets = tilesets.validate(pixel_format, palette.clone())?;
        layers.validate(&tilesets)?;

        //let framedata = self.framedata;
        let framedata = self
            .framedata
            .validate(&layers, pixel_format, palette.clone())?;

        Ok(ValidatedParseInfo {
            layers,
            tilesets,
            framedata,
            external_files: self.external_files,
            palette,
            tags: self.tags.unwrap_or_default(),
            frame_times: self.frame_times,
            sprite_user_data: self.sprite_user_data,
            slices: self.slices,
        })
    }

    fn modify_tags_frame_positions_with_cels_replacement_entry_and_get_delta_and_deleted_frames(&mut self, cels_replacement: &CelsReplacementEntry) -> (i32, Vec<u16>) {
        let mut from_frame = 0;
        let mut to_frame = 0;
        let mut offset = 0i32;
        let mut success = false;
        let mut deleted_frames: Vec<u16> = Vec::default();
        for mut tags in self.tags.iter_mut() {
            for mut tag in tags.iter_mut() {
                let tag_name = String::from(tag.name());
                if cels_replacement.name == tag_name {
                    from_frame = tag.from_frame();
                    to_frame = from_frame + cels_replacement.num_frames() - 1;
                    offset = ((cels_replacement.num_frames() - 1) as i32 - (tag.to_frame() - from_frame) as i32) as i32;

                    if offset < 0 {
                        // Push the frames to be deleted because you have shortened the animation and removed frames from the end.
                        for frame in ((offset + ((tag.to_frame() + 1) as i32)) as u32)..=tag.to_frame() {
                            deleted_frames.push(frame as u16);
                        }
                    }

                    tag.set_to_frame(to_frame);
                    success = true;
                }
            }
        }
        assert!(success, "modify_tags_frame_positions_with_cels_replacement_entry: Did not find tag in this cels_replacement_entry: {:?}", cels_replacement);

        for mut tags in self.tags.iter_mut() {
            for mut tag in tags.iter_mut() {
                let tag_name = String::from(tag.name());
                if cels_replacement.name == tag_name {
                    // do nothing
                } else {
                    if tag.from_frame() >= from_frame {
                        tag.set_from_frame(((tag.from_frame() as i32) + offset) as u32);
                    }
                    if tag.to_frame() >= to_frame {
                        tag.set_to_frame(((tag.to_frame() as i32) + offset) as u32);
                    }
                    assert!(tag.from_frame() <= tag.to_frame());
                }
            }
        }
        (offset, deleted_frames)
    }

    fn modify_tags_frame_positions_with_cels_replacements(
        &mut self,
        cels_replacements: &CelsReplacements,
        total_offset: &mut i32,
        total_deleted_frames: &mut Vec<u16>,
        extended_frames: &mut Vec<(String, u16)>
    ) {
        let mut total_deleted_frames: Vec<u16> = Vec::default();
        // First component of tuple is the from_frame
        let mut extended_frames: Vec<(String, u16)> = Vec::default();
        for cels_replacement in cels_replacements.data.iter() {
            let (offset, mut deleted_frames) = self.modify_tags_frame_positions_with_cels_replacement_entry_and_get_delta_and_deleted_frames(&cels_replacement);
            if offset > 0 {
                extended_frames.push((cels_replacement.name.clone(), offset as u16));
            }
            total_deleted_frames.append(&mut deleted_frames);
            *total_offset += offset;
        }
    }

    fn build_tags_chunk_data(&self) -> Vec<u8> {
        if let Some(tags) = &self.tags {
            let num_tags: u16 = tags.len() as u16;
            let mut builder = BytesBuilder::default();
            builder.push_word(num_tags);
            builder.push_bytes(vec![0u8; 8]);

            for tag in tags {
                builder.push_word(tag.from_frame() as u16);
                builder.push_word(tag.to_frame() as u16);
                builder.push_byte(match tag.animation_direction() {
                    AnimationDirection::Forward => 0u8,
                    AnimationDirection::Reverse => 1u8,
                    AnimationDirection::PingPong => 2u8,
                });
                builder.push_bytes(vec![0u8; 8]);
                builder.push_dword(0u32); // color
                builder.push_string(&tag.name());
            }

            builder.bytes
        } else {
            panic!("build_tags_chunk was called when tags was None. this shouldn't happen");
        }
    }

    fn frame_id_contains_replaceable_cels_in_entry(&self, frame_id: u16, cels_replacement: &CelsReplacementEntry) -> bool {
        let frame_id: u32 = frame_id as u32;
        for tag in self.tags.as_ref().unwrap() {
            let tag_name = String::from(tag.name());
            if cels_replacement.name == tag_name && frame_id >= tag.from_frame() && frame_id <= tag.to_frame() {
                return true;
            }
        }
        return false;
    }
    fn frame_id_contains_replaceable_cels(&self, frame_id: u16, cels_replacements: &CelsReplacements) -> bool {
        for cels_replacement in cels_replacements.data.iter() {
            if self.frame_id_contains_replaceable_cels_in_entry(frame_id, &cels_replacement) {
                return true;
            }
        }
        return false;
    }

    fn build_cel_chunk_data(&self, frame_id: u16, layer_index: u16, cel_layer_frame_data: &CelLayerFrameData) -> Vec<u8> {
        let CelLayerFrameData {
            image,
            x,
            y,
            opacity,
        } = cel_layer_frame_data;
        let cel_type = 0u16; // TODO how do I determine this, properly?
        let mut builder = BytesBuilder::default();
        builder.push_word(layer_index);
        builder.push_short(*x);
        builder.push_short(*y);
        builder.push_byte(*opacity);
        builder.push_word(cel_type);
        builder.push_bytes(vec![0u8; 7]);
        fn build_raw_cel(image: &image::RgbaImage) -> Vec<u8> {
            // parse_raw_cel(reader, pixel_format).map(CelContent::Raw),
            let mut builder = BytesBuilder::default();
            builder.push_word(image.width() as u16);
            builder.push_word(image.height() as u16);

            // convert image to bytes here
            builder.push_bytes(image.clone().into_raw());

            builder.bytes
        }
        fn build_compressed_cel(image: &image::RgbaImage) -> Vec<u8> {
            // TODO let mut encoder = ZlibEncoder::new(...);
            unimplemented!();
            build_raw_cel(image) // TODO nocheckin compress this
        }
        let bytes = match cel_type {
            0 => build_raw_cel(image),
            2 => build_compressed_cel(image),
            _ => panic!("Read invalid cel_type in build_cel_chunk_data (We only support cel_types 0 and 2 currently): the invalid cel_type that was received was {:?}", cel_type),
        };
        builder.push_bytes(bytes);
        builder.bytes
    }

    fn build_cel_chunks_for_frame_with_cel_replacements(&self, frame_id: u16, cels_replacements: &CelsReplacements) -> Vec<Chunk> {
        let mut chunks: Vec<Chunk> = Vec::default();
        let frame_id: u32 = frame_id as u32;
        for tag in self.tags.as_ref().unwrap() {
            let tag_name = String::from(tag.name());
            for cels_replacement in &cels_replacements.data {
                if cels_replacement.name == tag_name && tag.from_frame() <= frame_id && tag.to_frame() >= frame_id {
                    for (layer_index, cel_layer_frame_data) in cels_replacement.data[frame_id as usize - tag.from_frame() as usize].iter().enumerate() {
                        let mut builder = BytesBuilder::default();
                        builder.push_bytes(self.build_cel_chunk_data(frame_id as u16, layer_index as u16, cel_layer_frame_data));
                        let chunk = Chunk {
                            chunk_type: ChunkType::Cel,
                            data: builder.bytes,
                        };
                        chunks.push(chunk);
                    }
                }
            }
        }
        chunks
    }
}

struct ValidatedParseInfo {
    layers: layer::LayersData,
    tilesets: TilesetsById,
    framedata: cel::CelsData<Pixels>,
    external_files: ExternalFilesById,
    palette: Option<Arc<palette::ColorPalette>>,
    tags: Vec<Tag>,
    frame_times: Vec<u16>,
    sprite_user_data: Option<UserData>,
    slices: Vec<Slice>,
}

// file format docs: https://github.com/aseprite/aseprite/blob/master/docs/ase-file-specs.md
// v1.3 spec diff doc: https://gist.github.com/dacap/35f3b54fbcd021d099e0166a4f295bab
pub fn read_aseprite<R: Read>(input: R) -> Result<AsepriteFile> {
    let mut reader = AseReader::with(input);
    let _size = reader.dword()?;
    let magic_number = reader.word()?;
    if magic_number != 0xA5E0 {
        return Err(AsepriteParseError::InvalidInput(format!(
            "Invalid magic number for header: {:x} != {:x}",
            magic_number, 0xA5E0
        )));
    }

    let num_frames = reader.word()?;
    let width = reader.word()?;
    let height = reader.word()?;
    let color_depth = reader.word()?;
    let _flags = reader.dword()?;
    let default_frame_time = reader.word()?;
    let _placeholder1 = reader.dword()?;
    let _placeholder2 = reader.dword()?;
    let transparent_color_index = reader.byte()?;
    let _ignore1 = reader.byte()?;
    let _ignore2 = reader.word()?;
    let _num_colors = reader.word()?;
    let pixel_width = reader.byte()?;
    let pixel_height = reader.byte()?;
    let _grid_x = reader.short()?;
    let _grid_y = reader.short()?;
    let _grid_width = reader.word()?;
    let _grid_height = reader.word()?;
    reader.skip_reserved(84)?;

    if !(pixel_width == 1 && pixel_height == 1) {
        return Err(AsepriteParseError::UnsupportedFeature(
            "Only pixel width:height ratio of 1:1 supported".to_owned(),
        ));
    }

    let mut parse_info = ParseInfo::new(num_frames, default_frame_time);

    let pixel_format = parse_pixel_format(color_depth, transparent_color_index)?;

    for frame_id in 0..num_frames {
        // println!("--- Frame {} -------", frame_id);
        parse_frame(&mut reader, frame_id, pixel_format, &mut parse_info)?;
    }

    let ValidatedParseInfo {
        layers,
        tilesets,
        framedata,
        external_files,
        palette,
        tags,
        frame_times,
        sprite_user_data,
        slices,
    } = parse_info.validate(&pixel_format)?;

    Ok(AsepriteFile {
        width,
        height,
        num_frames,
        pixel_format,
        palette,
        layers,
        frame_times,
        tags,
        framedata,
        external_files,
        tilesets,
        sprite_user_data,
        slices,
    })
}

fn parse_frame<R: Read>(
    reader: &mut AseReader<R>,
    frame_id: u16,
    pixel_format: PixelFormat,
    parse_info: &mut ParseInfo,
) -> Result<()> {
    let num_bytes = reader.dword()?;
    let magic_number = reader.word()?;
    if magic_number != 0xF1FA {
        return Err(AsepriteParseError::InvalidInput(format!(
            "Invalid magic number for frame: {:x} != {:x}",
            magic_number, 0xF1FA
        )));
    }
    let old_num_chunks = reader.word()?;
    let frame_duration_ms = reader.word()?;
    let _placeholder = reader.word()?;
    let new_num_chunks = reader.dword()?;

    parse_info.frame_times[frame_id as usize] = frame_duration_ms;

    let num_chunks = if new_num_chunks == 0 {
        old_num_chunks as u32
    } else {
        new_num_chunks
    };

    let bytes_available = num_bytes as i64 - FRAME_HEADER_SIZE;

    let chunks = Chunk::read_all(num_chunks, bytes_available, reader)?;

    for chunk in chunks {
        let Chunk { chunk_type, data } = chunk;
        match chunk_type {
            ChunkType::ColorProfile => {
                let profile = color_profile::parse_chunk(&data)?;
                parse_info.color_profile = Some(profile);
            }
            ChunkType::Palette => {
                let palette = palette::parse_chunk(&data)?;
                parse_info.palette = Some(Arc::new(palette));
            }
            ChunkType::Layer => {
                let layer_data = layer::parse_chunk(&data)?;
                parse_info.add_layer(layer_data);
            }
            ChunkType::Cel => {
                let cel = cel::parse_chunk(&data, pixel_format)?;
                parse_info.add_cel(frame_id, cel)?;
            }
            ChunkType::ExternalFiles => {
                let files = ExternalFile::parse_chunk(&data)?;
                parse_info.add_external_files(files);
            }
            ChunkType::Tags => {
                let tags = tags::parse_chunk(&data)?;
                if frame_id == 0 {
                    parse_info.add_tags(tags);
                } else {
                    debug!("Ignoring tags outside of frame 0");
                }
            }
            ChunkType::Slice => {
                let slice = slice::parse_chunk(&data)?;
                parse_info.add_slice(slice);
            }
            ChunkType::UserData => {
                let user_data = user_data::parse_userdata_chunk(&data)?;
                parse_info.add_user_data(user_data)?;
            }
            ChunkType::OldPalette04 => {
                // An old palette chunk precedes the sprite UserData chunk.
                // Update the chunk context to reflect the OldPalette chunk.
                parse_info.user_data_context = Some(UserDataContext::OldPalette);

                if parse_info.palette.is_none() {
                    let palette = palette::parse_old_chunk_04(&data)?;
                    parse_info.palette = Some(Arc::new(palette));
                }
            }
            ChunkType::OldPalette11 => {
                // An old palette chunk precedes the sprite UserData chunk.
                // Update the chunk context to reflect the OldPalette chunk.
                parse_info.user_data_context = Some(UserDataContext::OldPalette);

                if parse_info.palette.is_none() {
                    let palette = palette::parse_old_chunk_11(&data)?;
                    parse_info.palette = Some(Arc::new(palette));
                }
            }
            ChunkType::Tileset => {
                let tileset = Tileset::<RawPixels>::parse_chunk(&data, pixel_format)?;
                parse_info.tilesets.add(tileset);
            }
            ChunkType::CelExtra | ChunkType::Mask | ChunkType::Path => {
                debug!("Ignoring unsupported chunk type: {:?}", chunk_type);
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum UserDataContext {
    CelId(CelId),
    LayerIndex(u32),
    OldPalette,
    TagIndex(u16),
    SliceIndex(u32),
}

#[derive(Debug, Clone, PartialEq)]
enum ChunkType {
    OldPalette04, // deprecated
    OldPalette11, // deprecated
    Palette,
    Layer,
    Cel,
    CelExtra,
    ColorProfile,
    Mask, // deprecated
    Path,
    Tags,
    UserData,
    Slice,
    ExternalFiles,
    Tileset,
}

fn parse_chunk_type(chunk_type: u16) -> Result<ChunkType> {
    match chunk_type {
        0x0004 => Ok(ChunkType::OldPalette04),
        0x0011 => Ok(ChunkType::OldPalette11),
        0x2004 => Ok(ChunkType::Layer),
        0x2005 => Ok(ChunkType::Cel),
        0x2006 => Ok(ChunkType::CelExtra),
        0x2007 => Ok(ChunkType::ColorProfile),
        0x2008 => Ok(ChunkType::ExternalFiles),
        0x2016 => Ok(ChunkType::Mask),
        0x2017 => Ok(ChunkType::Path),
        0x2018 => Ok(ChunkType::Tags),
        0x2019 => Ok(ChunkType::Palette),
        0x2020 => Ok(ChunkType::UserData),
        0x2022 => Ok(ChunkType::Slice),
        0x2023 => Ok(ChunkType::Tileset),
        _ => Err(AsepriteParseError::UnsupportedFeature(format!(
            "Invalid or unsupported chunk type: 0x{:x}",
            chunk_type
        ))),
    }
}

const CHUNK_HEADER_SIZE: usize = 6;
const FRAME_HEADER_SIZE: i64 = 16;

#[derive(Clone)]
pub struct Chunk {
    chunk_type: ChunkType,
    data: Vec<u8>,
}

impl Chunk {
    fn read<R: Read>(bytes_available: &mut i64, reader: &mut AseReader<R>) -> Result<Self> {
        let chunk_size = reader.dword()?;
        let chunk_type_code = reader.word()?;
        let chunk_type = parse_chunk_type(chunk_type_code)?;

        check_chunk_bytes(chunk_size, *bytes_available)?;

        let chunk_data_bytes = chunk_size as usize - CHUNK_HEADER_SIZE;
        let mut data = vec![0_u8; chunk_data_bytes];
        reader.read_exact(&mut data)?;
        *bytes_available -= chunk_size as i64;
        Ok(Chunk { chunk_type, data })
    }
    fn read_all<R: Read>(
        count: u32,
        mut bytes_available: i64,
        reader: &mut AseReader<R>,
    ) -> Result<Vec<Self>> {
        let mut chunks: Vec<Chunk> = Vec::new();
        for _idx in 0..count {
            let chunk = Self::read(&mut bytes_available, reader)?;
            chunks.push(chunk);
        }
        Ok(chunks)
    }
    fn get_header_bytes(&self) -> Vec<u8> {
        let mut builder = BytesBuilder::with_capacity(CHUNK_HEADER_SIZE as usize);
        let chunk_size = self.data.len() as u32 + CHUNK_HEADER_SIZE as u32;
        let chunk_type_code = match self.chunk_type {
            ChunkType::OldPalette04 => 0x0004,
            ChunkType::OldPalette11 => 0x0011,
            ChunkType::Layer => 0x2004,
            ChunkType::Cel => 0x2005,
            ChunkType::CelExtra => 0x2006,
            ChunkType::ColorProfile => 0x2007,
            ChunkType::ExternalFiles => 0x2008,
            ChunkType::Mask => 0x2016,
            ChunkType::Path => 0x2017,
            ChunkType::Tags => 0x2018,
            ChunkType::Palette => 0x2019,
            ChunkType::UserData => 0x2020,
            ChunkType::Slice => 0x2022,
            ChunkType::Tileset => 0x2023,
        };
        builder.push_dword(chunk_size);
        builder.push_word(chunk_type_code);
        builder.bytes
    }
}

fn check_chunk_bytes(chunk_size: u32, bytes_available: i64) -> Result<()> {
    if (chunk_size as usize) < CHUNK_HEADER_SIZE {
        return Err(AsepriteParseError::InvalidInput(format!(
            "Chunk size is too small {}, minimum_size: {}",
            chunk_size, CHUNK_HEADER_SIZE
        )));
    }
    if chunk_size as i64 > bytes_available {
        return Err(AsepriteParseError::InvalidInput(format!(
            "Trying to read chunk of size {}, but there are only {} bytes available in the frame",
            chunk_size, bytes_available
        )));
    }
    Ok(())
}

fn parse_pixel_format(color_depth: u16, transparent_color_index: u8) -> Result<PixelFormat> {
    match color_depth {
        8 => Ok(PixelFormat::Indexed {
            transparent_color_index,
        }),
        16 => Ok(PixelFormat::Grayscale),
        32 => Ok(PixelFormat::Rgba),
        _ => Err(AsepriteParseError::InvalidInput(format!(
            "Unknown pixel format. Color depth: {}",
            color_depth
        ))),
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct FrameHeader {
    num_bytes: u32,
    old_num_chunks: u16,
    frame_duration_ms: u16,
    new_num_chunks: u32,
}

impl FrameHeader {
    fn get_bytes(&self) -> Vec<u8> {
        let FrameHeader {
            old_num_chunks,
            frame_duration_ms,
            new_num_chunks,
            num_bytes,
        } = *self;
        let mut builder = BytesBuilder::with_capacity(FRAME_HEADER_SIZE as usize);
        builder.push_dword(num_bytes);
        let magic_number = 0xF1FA;
        builder.push_word(magic_number);
        builder.push_word(old_num_chunks);
        builder.push_word(frame_duration_ms);
        builder.push_word(0u16);
        builder.push_dword(new_num_chunks);
        builder.bytes
    }
}

/* TODO
This needs to contain the image information that I'm going to use to replace the cel animation with

I need all the insertions i'm going to make.

I need to know what each of the tags will be changed to.


 */

#[derive(Debug, Default)]
pub struct CelLayerFrameData {
    // frames to layers
    pub(crate) image: image::RgbaImage,
    pub(crate) x: i16,
    pub(crate) y: i16,
    pub(crate) opacity: u8,
}

#[derive(Debug)]
pub struct CelsReplacementEntry {
    // frames to layers
    pub(crate) data: Vec<Vec<CelLayerFrameData>>,
    pub(crate) name: String,
    pub(crate) default_frame_duration_ms: u16,
}

impl CelsReplacementEntry {
    fn num_frames(&self) -> u32 {
        self.data.len() as u32
    }
}

#[derive(Debug)]
pub struct CelsReplacements {
    pub(crate) data: Vec<CelsReplacementEntry>,
}

pub fn replace_animation_cels(path: PathBuf, cels_replacements: CelsReplacements) {
    let file = std::fs::File::open(&path).unwrap();
    let reader = std::io::BufReader::new(file);
    let bytes = get_aseprite_file_bytes_with_replaced_animation_cels(reader, cels_replacements).unwrap();
    std::fs::write(path, bytes).unwrap();
}

// file format docs: https://github.com/aseprite/aseprite/blob/master/docs/ase-file-specs.md
// v1.3 spec diff doc: https://gist.github.com/dacap/35f3b54fbcd021d099e0166a4f295bab
fn get_aseprite_file_bytes_with_replaced_animation_cels<R: Read>(input: R, cels_replacements: CelsReplacements) -> Result<Vec<u8>> {
    /*
    The goal of this function is to seek to the cels that I need to rewrite, then take the image data from the ase_file, and rewrite the image data for those cels.

    Maybe I could rewrite specific frames
    */

    let mut reader = AseReader::with(input);
    let mut builder = BytesBuilder::default();

    let size = reader.dword()?;
    builder.push_dword(size);
    let magic_number = reader.word()?;
    builder.push_word(magic_number);
    if magic_number != 0xA5E0 {
        return Err(AsepriteParseError::InvalidInput(format!(
            "Invalid magic number for header: {:x} != {:x}",
            magic_number, 0xA5E0
        )));
    }

    let mut num_frames = reader.word()?;

    // This value will be rewritten with correct value later, because it is not known yet until we apply the changes to the animation.
    builder.push_word(0u16);

    builder.push_dword(reader.dword()?); // size
    let color_depth = reader.word()?;
    builder.push_word(color_depth);
    builder.push_dword(reader.dword()?);
    let default_frame_time = reader.word()?;
    builder.push_word(default_frame_time);
    builder.push_dword(reader.dword()?);
    builder.push_dword(reader.dword()?);
    let transparent_color_index = reader.byte()?;
    builder.push_byte(transparent_color_index);
    builder.push_dword(reader.dword()?);
    builder.push_dword(reader.dword()?);
    builder.push_dword(reader.dword()?);
    builder.push_word(reader.word()?);
    builder.push_byte(reader.byte()?);
    reader.skip_reserved(84)?;
    builder.push_bytes(vec![0u8; 84]);

    assert_eq!(builder.size(), 128usize, "The bytes builder, which current is supposed to have built the header to the aseprite file, does not have the correct size");

    // TODO test extending the number of frames in an animation, because you need to modify parse info after you determining how much the total frame size will be changed
    let mut parse_info = ParseInfo::new(num_frames, default_frame_time);

    let pixel_format = parse_pixel_format(color_depth, transparent_color_index)?;

    // TODO for now we are assuming the pixel format is Rgba
    assert_eq!(pixel_format, PixelFormat::Rgba, "replace_animation_cels currently only supports images with PixelFormat::Rgba");

    let mut new_num_bytes = 128u32;
    let mut new_num_frames = 0u16;

    let mut offset_to_num_frames = 0i32;
    let mut old_frame_id_to_be_deleted: Vec<u16> = vec![];
    let mut extended_animations_heap: BinaryHeap<(u16, u16, u16)> = BinaryHeap::default();

    let mut chunks_by_frame: Vec<(FrameHeader, Vec<Chunk>)> = Vec::with_capacity(num_frames as usize);

    for frame_id in 0..num_frames {
        let (mut frame_header, mut chunks) = parse_frame_header_and_chunks_for_replacing_animation(&mut reader, frame_id, pixel_format, &mut parse_info)?;

        if frame_id == 0 {
            for mut chunk in chunks.iter_mut() {
                if chunk.chunk_type == ChunkType::Tags {
                    // Now I need to offset all the from_frames and to_frames.
                    // Apply the offsets to the frame positions in the parse_info's tag data
                    let mut extended_animations: Vec<(String, u16)> = Vec::default();

                    parse_info.modify_tags_frame_positions_with_cels_replacements(&cels_replacements, &mut offset_to_num_frames, &mut old_frame_id_to_be_deleted, &mut extended_animations);

                    for (tag_name, amount_extended) in extended_animations.iter() {
                        let mut from_frame = 0u32;
                        let mut default_frame_duration_ms = 0u16;
                        for tags in parse_info.tags.iter() {
                            for tag in tags.iter() {
                                for cels_replacement in &cels_replacements.data {
                                    if cels_replacement.name == *tag_name {
                                        default_frame_duration_ms = cels_replacement.default_frame_duration_ms;
                                    }
                                }
                                if tag_name == tag.name() {
                                    from_frame = tag.from_frame();
                                }
                            }
                        }
                        extended_animations_heap.push((from_frame as u16, *amount_extended, default_frame_duration_ms));
                    }
                    assert!(parse_info.tags.is_some(), "Error occured in `replace_animation_cels`: This aseprite file contains no tags. We cannot replace any animations for an aseprite file that has no tags.");

                    new_num_frames = ((num_frames as i32) + offset_to_num_frames) as u16;

                    // Add the offset to num frames
                    chunk.data = parse_info.build_tags_chunk_data();
                }
            }
        }

        chunks_by_frame.push((frame_header, chunks));
    }

    // Remove frames due to shortening size of an animation
    let mut frame_id = 0;
    chunks_by_frame.retain(|chunks_by_frame| {
        let is_retain = !old_frame_id_to_be_deleted.contains(&frame_id);
        frame_id += 1;
        is_retain
    });

    // Now insert frames to extended animations, frame_duration_ms is derived from default_frame_duration_ms from the `cels_replacement` input parameter.
    while extended_animations_heap.len() > 0 {
        if let Some((from_frame, amount_extended, default_frame_duration_ms)) = extended_animations_heap.pop() {
            // Insert blank frames into animation which is getting extended
            // The cel chunks will be added later
            for _ in 0..amount_extended {
                chunks_by_frame.insert(
                    from_frame as usize,
                    (
                        FrameHeader {
                            frame_duration_ms: default_frame_duration_ms,
                            ..Default::default()
                        },
                        vec![],
                    )
                );
            }
        }
    }

    let mut frame_id = 0;
    for (frame_header, chunks) in chunks_by_frame.iter_mut() {
        // Remove cel chunks if this is a frame in the cels_replacemnet
        if parse_info.frame_id_contains_replaceable_cels(frame_id, &cels_replacements) {
            chunks.retain(|chunk| {
                chunk.chunk_type != ChunkType::Cel
            });
        }

        // insert replacement Cel chunks for this frame
        for cel_chunk in parse_info.build_cel_chunks_for_frame_with_cel_replacements(frame_id, &cels_replacements) {
            chunks.push(cel_chunk);
        }

        // modify frame_header with reduced num_bytes and num_chunks
        fn get_number_of_bytes_from_chunks(chunks: &Vec<Chunk>) -> u32 {
            fn get_chunk_len(chunk: &Chunk) -> u32 {
                (chunk.data.len() as u32) + (CHUNK_HEADER_SIZE as u32)
            }
            let mut num_bytes = 0;
            for chunk in chunks {
                num_bytes += get_chunk_len(chunk);
            }
            num_bytes
        }

        frame_header.num_bytes = (FRAME_HEADER_SIZE as u32) + get_number_of_bytes_from_chunks(&chunks);
        frame_header.old_num_chunks = chunks.len() as u16;
        frame_header.new_num_chunks = chunks.len() as u32;
        new_num_bytes += frame_header.num_bytes;
        frame_id += 1;
    }

    builder.set_dword_at_offset(0, new_num_bytes);
    builder.set_word_at_offset(6, new_num_frames);

    // Now append all the chunks, both the ones parsed from the original file, with the changes, insertions, and deletions of chunks we made, into the bytes builder
    for (frame_header, chunks) in chunks_by_frame {
        builder.push_bytes(frame_header.get_bytes());
        for chunk in chunks {
            builder.push_bytes(chunk.get_header_bytes());
            builder.push_bytes(chunk.data);
        }
    }

    //// TODO Maybe I need to remodify the parse_info when I insert the new cel chunks, then validate it again
    //let ValidatedParseInfo {
    //    layers,
    //    tilesets,
    //    framedata,
    //    external_files,
    //    palette,
    //    tags,
    //    frame_times,
    //    sprite_user_data,
    //    slices,
    //} = parse_info.validate(&pixel_format)?;

    // TODO assert the file_size is correct

    Ok(builder.bytes)
}

fn parse_frame_header_and_chunks_for_replacing_animation<R: Read>(
    reader: &mut AseReader<R>,
    frame_id: u16,
    pixel_format: PixelFormat,
    parse_info: &mut ParseInfo,
) -> Result<(FrameHeader, Vec<Chunk>)> {
    let num_bytes = reader.dword()?;
    let magic_number = reader.word()?;
    if magic_number != 0xF1FA {
        return Err(AsepriteParseError::InvalidInput(format!(
            "Invalid magic number for frame: {:x} != {:x}",
            magic_number, 0xF1FA
        )));
    }
    let old_num_chunks = reader.word()?;
    let frame_duration_ms = reader.word()?;
    let _placeholder = reader.word()?;
    let new_num_chunks = reader.dword()?;

    parse_info.frame_times[frame_id as usize] = frame_duration_ms;

    let frame_header = FrameHeader {
        old_num_chunks,
        frame_duration_ms,
        new_num_chunks,
        num_bytes,
    };

    let num_chunks = if new_num_chunks == 0 {
        old_num_chunks as u32
    } else {
        new_num_chunks
    };

    let bytes_available = num_bytes as i64 - FRAME_HEADER_SIZE;

    let chunks = Chunk::read_all(num_chunks, bytes_available, reader)?;

    for chunk in &chunks {
        let Chunk { chunk_type, data } = chunk;
        match chunk_type {
            ChunkType::Tags => {
                let tags = tags::parse_chunk(&data)?;
                if frame_id == 0 {
                    parse_info.add_tags(tags);
                } else {
                    debug!("Ignoring tags outside of frame 0");
                }
            }
            _ => {
                debug!("Ignoring unsupported chunk type when reading file that is to have selected animations replaced. chunk_type is: {:?}", chunk_type);
            }
        }
    }

    Ok((frame_header, chunks))
}
