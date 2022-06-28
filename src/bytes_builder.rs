#[derive(Default)]
pub(crate) struct BytesBuilder {
    pub bytes: Vec<u8>,
}

impl BytesBuilder {
    pub(crate) fn with_capacity(size: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(size),
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn set_byte_at_offset(&mut self, byte_offset: usize, input: u8) {
        self.bytes[byte_offset] = input;
    }

    pub(crate) fn set_word_at_offset(&mut self, byte_offset: usize, input: u16) {
        self.bytes[byte_offset] = (input & 0xff) as u8;
        self.bytes[byte_offset + 1] = ((input >> 8) & 0xff) as u8;
    }

    pub(crate) fn set_dword_at_offset(&mut self, byte_offset: usize, input: u32) {
        self.bytes[byte_offset] = (input & 0xff) as u8;
        self.bytes[byte_offset + 1] = ((input >> 8) & 0xff) as u8;
        self.bytes[byte_offset + 2] = ((input >> 16) & 0xff) as u8;
        self.bytes[byte_offset + 3] = ((input >> 24) & 0xff) as u8;
    }

    pub(crate) fn push_byte(&mut self, input: u8) {
        self.bytes.push(input);
    }

    pub(crate) fn push_word(&mut self, input: u16) {
        self.bytes.push((input & 0xff) as u8);
        self.bytes.push(((input >> 8) & 0xff) as u8);
    }

    pub(crate) fn push_short(&mut self, input: i16) {
        self.bytes.push((input & 0xff) as u8);
        self.bytes.push(((input >> 8) & 0xff) as u8);
    }

    pub(crate) fn push_dword(&mut self, input: u32) {
        self.bytes.push((input & 0xff) as u8);
        self.bytes.push(((input >> 8) & 0xff) as u8);
        self.bytes.push(((input >> 16) & 0xff) as u8);
        self.bytes.push(((input >> 24) & 0xff) as u8);
    }

    pub(crate) fn push_long(&mut self, input: u64) {
        self.bytes.push((input & 0xff) as u8);
        self.bytes.push(((input >> 8) & 0xff) as u8);
        self.bytes.push(((input >> 16) & 0xff) as u8);
        self.bytes.push(((input >> 24) & 0xff) as u8);
        self.bytes.push(((input >> 32) & 0xff) as u8);
        self.bytes.push(((input >> 40) & 0xff) as u8);
        self.bytes.push(((input >> 48) & 0xff) as u8);
        self.bytes.push(((input >> 56) & 0xff) as u8);
    }

    pub(crate) fn push_string(&mut self, string: &str) {
        self.push_word(string.len() as u16);
        for byte in &string.as_bytes()[0..string.len()] {
            self.push_byte(*byte);
        }
    }

    pub(crate) fn push_bytes(&mut self, bytes: Vec<u8>) {
        for byte in bytes {
            self.push_byte(byte);
        }
    }
}
