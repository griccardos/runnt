use ndarray::Array1;

use crate::{error::Error, sede::Sede};

#[derive(Debug, PartialEq)]
pub(crate) struct Dropout {
    rate: f32,
    values: Array1<f32>,
}
impl Dropout {
    pub(crate) fn new(amount: f32, size: usize) -> Self {
        assert!(
            amount > 0. && amount < 1.,
            "Dropout amount must be in range (0,1)"
        );
        let values = Array1::zeros(size);
        Self {
            rate: amount,
            values,
        }
    }
    pub(crate) fn recalc(&mut self) {
        self.values.mapv_inplace(|_|
                //keep with prob 1-dropout
                if fastrand::f32() < self.rate {
                    0.
                } else {
                    1. / (1. - self.rate) //scale up to keep expected value the same
                }
            );
    }
    pub(crate) fn mask(&self) -> Array1<f32> {
        self.values.clone()
    }
}
impl Sede for Option<Dropout> {
    fn serialize(&self) -> String {
        if let Some(d) = self {
            format!("{}:{}", d.rate, d.values.len())
        } else {
            "None".to_string()
        }
    }

    fn deserialize(s: &str) -> Result<Self, Error>
    where
        Self: Sized,
    {
        if s == "None" {
            return Ok(None);
        }
        if let Some((rate_str, size_str)) = s.split_once(':') {
            let rate: f32 = rate_str
                .parse()
                .map_err(|_| Error::ParseError("Invalid dropout rate".to_string()))?;
            let size: usize = size_str
                .parse()
                .map_err(|_| Error::ParseError("Invalid dropout size".to_string()))?;
            Ok(Some(Dropout::new(rate, size)))
        } else {
            Err(Error::ParseError("Invalid dropout format".to_string()))
        }
    }
}
