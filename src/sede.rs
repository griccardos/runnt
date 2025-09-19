use ndarray::Array2;

use crate::error::Error;

pub trait Sede {
    fn serialize(&self) -> String;
    fn deserialize(s: &str) -> Result<Self, Error>
    where
        Self: Sized;
}

impl Sede for Array2<f32> {
    fn serialize(&self) -> String {
        self.rows()
            .into_iter()
            .map(|x| {
                x.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join(";")
    }

    fn deserialize(s: &str) -> Result<Self, Error> {
        if s.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let rows = s
            .split(';')
            .map(|row| {
                row.split(',')
                    .map(|val| val.parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| Error::SerializationError(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if rows.is_empty() || rows[0].is_empty() {
            return Err(Error::SerializationError("Empty array".to_string()));
        }

        let cols = rows[0].len();
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((flat.len() / cols, cols), flat)
            .map_err(|e| Error::SerializationError(e.to_string()))
    }
}

impl Sede for Vec<Array2<f32>> {
    fn serialize(&self) -> String {
        self.iter()
            .map(|arr| arr.serialize())
            .collect::<Vec<_>>()
            .join("|")
    }

    fn deserialize(s: &str) -> Result<Self, Error> {
        s.split('|')
            .map(|part| Array2::deserialize(part))
            .collect::<Result<Vec<_>, _>>()
    }
}
