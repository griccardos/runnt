use std::fmt::Display;

pub enum Regularization {
    None,
    ///try 0.00001
    L1(f32),
    ///try 0.0001
    L2(f32),
    ///try 0.00001, 0.0001
    L1L2(f32, f32),
}
impl Regularization {
    ///Uses default of 0.00001
    pub fn l1() -> Regularization {
        Regularization::L1(0.00001)
    }
    ///Uses default of 0.0001
    pub fn l2() -> Regularization {
        Regularization::L2(0.0001)
    }
    ///Uses defaults of 0.00001, 0.0001
    pub fn l1l2() -> Regularization {
        Regularization::L1L2(0.00001, 0.0001)
    }

    pub(crate) fn from_str(line: &str) -> Regularization {
        if line == "None" {
            return Regularization::None;
        }

        if let Some((t, vs)) = line.split_once(':') {
            return match (t, vs) {
                ("L1", vs) => Regularization::L1(vs.parse::<f32>().unwrap_or_default()),
                ("L2", vs) => Regularization::L2(vs.parse::<f32>().unwrap_or_default()),
                ("L1L2", vs) => {
                    if let Some((v1, v2)) = vs.split_once(',') {
                        Regularization::L1L2(
                            v1.parse::<f32>().unwrap_or_default(),
                            v2.parse::<f32>().unwrap_or_default(),
                        )
                    } else {
                        Regularization::None
                    }
                }

                _ => Regularization::None,
            };
        }
        Regularization::None
    }
}

impl Display for Regularization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Regularization::None => write!(f, "None"),
            Regularization::L1(v) => write!(f, "L1:{v}"),
            Regularization::L2(v) => write!(f, "L2:{v}"),
            Regularization::L1L2(v1, v2) => write!(f, "L1L2:{v1},{v2}"),
        }
    }
}
