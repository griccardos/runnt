use std::num::ParseFloatError;

#[derive(Debug)]
pub enum Error {
    ParseError(String),
    SerializationError(String),
}

impl From<ParseFloatError> for Error {
    fn from(err: ParseFloatError) -> Self {
        Error::ParseError(err.to_string())
    }
}
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::SerializationError(err.to_string())
    }
}
