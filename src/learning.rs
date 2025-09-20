use std::fmt::Display;

use crate::{error::Error, sede::Sede};

pub struct LearningRate {
    pub rate: Rate,
    pub step: usize,
}
impl From<f32> for LearningRate {
    fn from(rate: f32) -> Self {
        Self::new(Rate::Constant(rate))
    }
}

impl LearningRate {
    pub fn new(rate: Rate) -> Self {
        Self { rate, step: 0 }
    }

    pub fn get(&self) -> f32 {
        match self.rate {
            Rate::Constant(lr) => lr,
            Rate::Cosine {
                start_rate,
                warmup_target_rate,
                warmup_steps,
                total_steps,
                min_rate,
            } => {
                let step = self.step as f32;
                let total_steps = total_steps as f32;
                let warmup_steps = warmup_steps as f32;
                let warmup_target_rate = if warmup_target_rate > 0. {
                    warmup_target_rate
                } else {
                    start_rate
                };
                assert!(warmup_target_rate >= min_rate);

                if step < warmup_steps {
                    return start_rate + (warmup_target_rate - start_rate) * (step / warmup_steps);
                }
                if step >= total_steps {
                    return min_rate;
                }

                let progress = (step - warmup_steps) / (total_steps - warmup_steps);
                let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                let decayed = (warmup_target_rate - min_rate) * cosine_decay + min_rate;
                decayed
            }
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
    }
}

impl Display for LearningRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant_name = match self.rate {
            Rate::Constant(_) => "Constant",
            Rate::Cosine { .. } => "Cosine",
        };
        write!(f, "{} {}", variant_name, self.get())
    }
}

pub enum Rate {
    Constant(f32),
    Cosine {
        start_rate: f32,
        warmup_target_rate: f32,
        warmup_steps: usize,
        ///this is the total number of steps (1 step is a batch update)
        total_steps: usize,
        min_rate: f32,
    },
}
impl Sede for LearningRate {
    fn serialize(&self) -> String {
        format!("step:{},rate:{}", self.step, self.rate.serialize())
    }

    fn deserialize(s: &str) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let mut parts = s.split(",rate:");
        let step_part = parts
            .next()
            .ok_or(Error::SerializationError("Missing step".into()))?;
        let rate_part = parts
            .next()
            .ok_or(Error::SerializationError("Missing rate".into()))?;

        let step_str = step_part
            .strip_prefix("step:")
            .ok_or(Error::SerializationError("Invalid step format".into()))?;
        let step = step_str
            .parse::<usize>()
            .map_err(|e| Error::SerializationError(e.to_string()))?;

        let rate = Rate::deserialize(rate_part)?;

        Ok(LearningRate { rate, step })
    }
}
impl Sede for Rate {
    fn serialize(&self) -> String {
        match self {
            Rate::Constant(lr) => format!("Constant:{}", lr),
            Rate::Cosine {
                start_rate,
                warmup_target_rate,
                warmup_steps,
                total_steps,
                min_rate,
            } => format!(
                "Cosine:start_rate={},warmup_target_rate={},warmup_steps={},total_steps={},min_rate={}",
                start_rate, warmup_target_rate, warmup_steps, total_steps, min_rate
            ),
        }
    }
    fn deserialize(s: &str) -> Result<Self, Error>
    where
        Self: Sized,
    {
        if s.starts_with("Constant:") {
            let lr_str = &s["Constant:".len()..];
            let lr = lr_str
                .parse::<f32>()
                .map_err(|e| Error::SerializationError(e.to_string()))?;
            Ok(Rate::Constant(lr))
        } else if s.starts_with("Cosine:") {
            let params_str = &s["Cosine:".len()..];
            let mut start_rate = None;
            let mut warmup_target_rate = None;
            let mut warmup_steps = None;
            let mut total_steps = None;
            let mut min_rate = None;

            for param in params_str.split(',') {
                let mut key_value = param.split('=');
                let key = key_value.next().unwrap_or("");
                let value = key_value.next().unwrap_or("");

                match key {
                    "start_rate" => {
                        start_rate = Some(
                            value
                                .parse::<f32>()
                                .map_err(|e| Error::SerializationError(e.to_string()))?,
                        )
                    }
                    "warmup_target_rate" => {
                        warmup_target_rate = Some(
                            value
                                .parse::<f32>()
                                .map_err(|e| Error::SerializationError(e.to_string()))?,
                        )
                    }
                    "warmup_steps" => {
                        warmup_steps = Some(
                            value
                                .parse::<usize>()
                                .map_err(|e| Error::SerializationError(e.to_string()))?,
                        )
                    }
                    "total_steps" => {
                        total_steps = Some(
                            value
                                .parse::<usize>()
                                .map_err(|e| Error::SerializationError(e.to_string()))?,
                        )
                    }
                    "min_rate" => {
                        min_rate = Some(
                            value
                                .parse::<f32>()
                                .map_err(|e| Error::SerializationError(e.to_string()))?,
                        )
                    }
                    _ => return Err(Error::SerializationError(format!("Unknown key: {}", key))),
                }
            }

            Ok(Rate::Cosine {
                start_rate: start_rate
                    .ok_or(Error::SerializationError("Missing start_rate".into()))?,
                warmup_target_rate: warmup_target_rate.ok_or(Error::SerializationError(
                    "Missing warmup_target_rate".into(),
                ))?,
                warmup_steps: warmup_steps
                    .ok_or(Error::SerializationError("Missing warmup_steps".into()))?,
                total_steps: total_steps
                    .ok_or(Error::SerializationError("Missing total_steps".into()))?,
                min_rate: min_rate.ok_or(Error::SerializationError("Missing min_rate".into()))?,
            })
        } else {
            Err(Error::SerializationError("Invalid format".into()))
        }
    }
}

mod tests {

    #[test]
    fn test_sede() {
        use super::*;
        let lr = LearningRate::new(Rate::Cosine {
            start_rate: 0.0,
            warmup_target_rate: 0.1,
            warmup_steps: 100,
            total_steps: 10000,
            min_rate: 0.001,
        });
        let s = lr.serialize();
        let lr2 = LearningRate::deserialize(&s).unwrap();
        assert_eq!(lr.serialize(), lr2.serialize());
        let lr3 = LearningRate::new(Rate::Constant(0.01));
        let s3 = lr3.serialize();
        let lr4 = LearningRate::deserialize(&s3).unwrap();
        assert_eq!(lr3.serialize(), lr4.serialize());
    }
    #[test]
    fn test_learning_rate() {
        use super::*;
        let mut lr = LearningRate::new(Rate::Cosine {
            start_rate: 0.0,
            warmup_target_rate: 0.1,
            warmup_steps: 10,
            total_steps: 100,
            min_rate: 0.001,
        });

        for step in 0..110 {
            lr.step = step;
            let rate = lr.get();
            if step < 10 {
                assert!(rate <= 0.1);
                assert!(rate >= 0.0);
            }
            if step >= 10 && step <= 100 {
                assert!(rate <= 0.1);
                assert!(rate >= 0.001);
            }
            if step > 100 {
                assert_eq!(rate, 0.001);
            }
        }
    }
}
