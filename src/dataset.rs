use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::RangeInclusive,
    path::Path,
};

type Inputs = Vec<f32>;
type Targets = Vec<f32>;

/// Holds and manages dataset
/// This is not required, as NN takes Vec<f32> directly
/// However you may want to manage the dataset for example by:
/// - ignoring certain columns
/// - normalising data
/// - one hot encoding
/// - read from csv
/// - manage test
///
/// Builder requires:
/// - Data - either from `read_csv` or `add_data`
/// - Input Columns
/// - Target Columns
///
/// Other methods are optional
///
/// Limited by size of memory, as all data is read to memory
///
/// ```rust
///   use runnt::dataset::Dataset;
///   let data=vec![vec![0,1,2,3,4,5,6,7,8,9,10,11,12,13],vec![0,1,2,3,4,5,6,7,8,9,10,11,12,13]];
///   let mut set = Dataset::builder()
///   .add_data(data)
///   .allocate_to_test_data(0.5)
///   .add_target_columns(&[13], runnt::dataset::Adjustment::F32)
///   .add_input_columns(
///       &[0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12],
///       runnt::dataset::Adjustment::NormaliseMean,
///   )
///   .add_input_columns(&[3, 8], runnt::dataset::Adjustment::OneHot)
///   .build();
/// ```
pub struct Dataset {
    data: Vec<(Inputs, Targets)>,      //(inputs, targets)
    test_data: Vec<(Inputs, Targets)>, //(inputs, targets)
    input_labels: Vec<String>,
    target_labels: Vec<String>,
}

impl Dataset {
    pub fn builder() -> DatasetBuilder {
        DatasetBuilder::new()
    }

    /// Selects a point at random
    pub fn get_one(&self) -> (&Inputs, &Targets) {
        let item = &self.data[fastrand::usize(0..self.data.len())];
        (item.0.as_ref(), item.1.as_ref())
    }
    /// Selects a test point at random
    pub fn get_one_test(&self) -> (&Inputs, &Targets) {
        let item = &self.test_data[fastrand::usize(0..self.test_data.len())];
        (item.0.as_ref(), item.1.as_ref())
    }

    /// Returns shuffled data into a Vec of inputs, and a Vec of targets
    /// easier for forward and fit
    pub fn get_data(&self) -> (Vec<&Inputs>, Vec<&Targets>) {
        self.get_shuffled_data(&self.data)
    }
    // returns shuffled data into a zipped Vec of (Input,Output)
    // easier for reporting
    pub fn get_data_zip(&self) -> Vec<(&Inputs, &Targets)> {
        let (inp, out) = self.get_data();
        inp.into_iter().zip(out).collect()
    }

    /// Returns shuffled test data
    pub fn get_test_data(&self) -> (Vec<&Inputs>, Vec<&Targets>) {
        self.get_shuffled_data(&self.test_data)
    }

    // returns shuffled test data into a zipped Vec of (Input,Output)
    // easier for reporting
    pub fn get_test_data_zip(&self) -> Vec<(&Inputs, &Targets)> {
        let (inp, out) = self.get_data();
        inp.into_iter().zip(out).collect()
    }

    fn get_shuffled_data(&self, data: &Vec<(Inputs, Targets)>) -> (Vec<&Inputs>, Vec<&Targets>) {
        let mut indices = (0..data.len()).into_iter().collect::<Vec<_>>();
        fastrand::shuffle(&mut indices);
        let mut vecin = vec![];
        let mut vectar = vec![];
        for i in indices {
            vecin.push(&self.data[i].0);
            vectar.push(&self.data[i].1);
        }

        (vecin, vectar)
    }

    pub fn input_size(&self) -> usize {
        self.data[0].0.len()
    }

    pub fn target_size(&self) -> usize {
        self.data[0].1.len()
    }

    pub fn input_labels(&self) -> &Vec<String> {
        &self.input_labels
    }

    pub fn target_labels(&self) -> &Vec<String> {
        &self.target_labels
    }
}
#[derive(Clone, Copy)]
///Adjustments available to convert a string to an f32<br>
/// `F32` - Converts string to f32 <br>
/// `NormaliseMean` - Normalises based on field values (x-mean)/stddev <br/>
/// `NormaliseMinMax(f32, f32)` - Nomalises between a given lower and upper bound <br/>
/// `OneHot` - Creates new column for each one of the unique values with 1 for that and 0 for others <br/>
/// `Function(fn(&str) -> f32)` - Applies function to convert to f32 <br/>

pub enum Adjustment {
    ///Converts string to f32
    F32,
    ///Normalises based on field values (x-mean)/stddev
    NormaliseMean,
    /// Nomalises between a given lower and upper bound
    NormaliseMinMax(f32, f32),
    /// Creates new column for each one of the unique values with 1 for that and 0 for others
    OneHot,
    ///Applies function to convert to f32
    Function(fn(&str) -> f32),
}

impl Debug for Adjustment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = match self {
            Adjustment::F32 => "F32",
            Adjustment::NormaliseMean => "NormaliseMean",
            Adjustment::NormaliseMinMax(_, _) => "NormaliseMinMax",
            Adjustment::OneHot => "OneHot",
            Adjustment::Function(_) => "Function",
        };
        write!(f, "{val}")
    }
}

#[derive(PartialEq, Eq, Debug)]
enum Location {
    Input,
    Target,
}

#[derive(Debug)]
struct Column {
    index: usize,
    adjustment: Adjustment,
    location: Location,
}

#[derive(Default)]
pub struct DatasetBuilder {
    data: Vec<Vec<String>>,
    test_data: Vec<Vec<String>>,
    columns: Vec<Column>,
    headers: Vec<String>,
}

impl DatasetBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// basic csv reading, for complex csv, or more options, use the `csv` crate
    /// Autodetects separator
    /// Assumes first row is column names, which it uses as headers
    pub fn read_csv(self, path: impl AsRef<Path>) -> Self {
        let data = std::fs::read_to_string(path).expect("Could not open csv");
        let mut seps: HashMap<char, usize> =
            HashMap::from_iter([(',', 0usize), ('\t', 0), (';', 0), ('|', 0)]);
        for char in data.chars() {
            if seps.contains_key(&char) {
                seps.entry(char).and_modify(|x| *x += 1);
            }
        }
        let sep = *seps
            .iter()
            .max_by(|a, b| a.1.cmp(b.1))
            .expect("hashmap should not be empty")
            .0;
        let data = data
            .as_str()
            .lines()
            .map(|line| {
                line.split(sep)
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
            })
            .collect::<Vec<_>>();
        let headers = data[0].clone();
        let data = data[1..].to_vec();

        self.add_column_headers(headers).add_data(data)
    }

    pub fn add_data<T: ToString>(mut self, data: Vec<Vec<T>>) -> Self {
        self.data.extend(
            data.into_iter()
                .map(|x| x.into_iter().map(|y| y.to_string()).collect()),
        );
        self
    }

    pub fn add_test_data<T: ToString>(mut self, test_data: Vec<Vec<T>>) -> Self {
        self.test_data.extend(
            test_data
                .into_iter()
                .map(|x| x.into_iter().map(|y| y.to_string()).collect()),
        );
        self
    }

    pub fn add_column_headers(mut self, headers: Vec<String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Allocates a percentage from train data to test data
    /// Data must already be added
    /// Randomly selects data to allocate
    /// If you'd prefer, you can also use `add_test_data` to add it directly
    /// <br>e.g. `allocate_to_test_data(0.2)` will allocate 20% of the data at random to test data
    pub fn allocate_to_test_data(mut self, ratio: f32) -> Self {
        let count = (self.data.len() as f32 * ratio) as usize;
        assert!(count > 0, "Not enough data to allocate to test data");
        let mut indices = (0..self.data.len()).into_iter().collect::<Vec<_>>();
        fastrand::shuffle(&mut indices);
        let mut indices: Vec<usize> = indices.iter().take(count).copied().collect();
        indices.sort();
        indices.reverse();
        for i in indices {
            self.test_data.push(self.data.remove(i));
        }
        self
    }

    /// Allocates a range from train data to test data.
    /// Data must already be added
    /// This selects the same training data each time
    /// <br>e.g `.allocate_range_to_test_data(0..=100)`
    pub fn allocate_range_to_test_data(mut self, range: RangeInclusive<usize>) -> Self {
        assert!(
            !self.data.is_empty(),
            "Not enough data to allocate to test data"
        );

        for i in range {
            self.test_data.push(self.data.remove(i));
        }
        self
    }

    /// Add inputs from columns zero indexed
    /// e.g. `.add_input_columns(&[0,4],Adjustment::F32)`
    pub fn add_input_columns(mut self, indexes: &[usize], adjustment: Adjustment) -> Self {
        for &index in indexes {
            self.columns.push(Column {
                index,
                adjustment,
                location: Location::Input,
            });
        }
        self
    }
    /// Add inputs from columns as zero indexed range
    /// e.g. `.add_input_columns(0..=4,Adjustment::F32)`
    pub fn add_input_columns_range(
        mut self,
        range: RangeInclusive<usize>,
        adjustment: Adjustment,
    ) -> Self {
        for index in range {
            self = self.add_input_columns(&[index], adjustment);
        }
        self
    }

    pub fn add_target_columns(mut self, indexes: &[usize], adjustment: Adjustment) -> Self {
        for &index in indexes {
            self.columns.push(Column {
                index,
                adjustment,
                location: Location::Target,
            });
        }
        self
    }

    pub fn build(&self) -> Dataset {
        self.asserts();

        let col_stats = ColumnStats {
            mean_sd: self.get_mean_sd(),
            min_max: self.get_min_max(),
            one_hot: self.get_one_hot(),
        };

        let input_labels = self.get_labels(&col_stats, Location::Input);
        let target_labels = self.get_labels(&col_stats, Location::Target);
        let (traininputs, traintargets) = self.transform(&self.data, &col_stats);
        let (testinputs, testtargets) = self.transform(&self.test_data, &col_stats);

        Dataset {
            data: traininputs
                .into_iter()
                .zip(traintargets)
                .map(|(i, o)| (i, o))
                .collect(),
            test_data: testinputs
                .into_iter()
                .zip(testtargets)
                .map(|(i, o)| (i, o))
                .collect(),
            input_labels,
            target_labels,
        }
    }

    fn get_labels(&self, cstats: &ColumnStats, location: Location) -> Vec<String> {
        let headers: Vec<String> = if self.headers.is_empty() {
            self.data[0]
                .iter()
                .enumerate()
                .map(|(i, _)| format!("Col{i}"))
                .collect()
        } else {
            self.headers.clone()
        };

        let mut labels = vec![];

        for col in &self.columns {
            if col.location == location {
                match col.adjustment {
                    Adjustment::OneHot => {
                        let mut strings: Vec<String> =
                            cstats.one_hot[&col.index].keys().cloned().collect();
                        //we store in alphabetical
                        strings.sort_by_key(|a| a.to_lowercase());
                        strings
                            .iter_mut()
                            .for_each(|x| *x = format!("{}_{x}", headers[col.index]));
                        labels.extend(strings);
                    }
                    _ => labels.push(headers[col.index].clone()),
                }
            }
        }
        labels
    }

    fn transform(
        &self,
        data: &Vec<Vec<String>>,
        cstats: &ColumnStats,
    ) -> (Vec<Inputs>, Vec<Targets>) {
        let mut indata = vec![];
        let mut outdata = vec![];

        for line in data {
            let mut newin = vec![];
            let mut newout = vec![];
            for col in &self.columns {
                let val = &line[col.index];
                let newval: Vec<f32> = match col.adjustment {
                    Adjustment::F32 => vec![val.parse::<f32>().unwrap_or_default()],
                    Adjustment::NormaliseMean => {
                        let val = val.parse::<f32>().unwrap_or_default();
                        let mean = cstats.mean_sd[&col.index].0;
                        let sd = cstats.mean_sd[&col.index].1.max(0.001); //must not be 0
                        vec![(val - mean) / sd]
                    }
                    Adjustment::OneHot => cstats.one_hot[&col.index][val].clone(),
                    Adjustment::Function(f) => vec![f(val)],
                    Adjustment::NormaliseMinMax(lower, upper) => {
                        let val = val.parse::<f32>().unwrap_or_default();
                        let min = cstats.min_max[&col.index].0;
                        let max = cstats.min_max[&col.index].1;
                        vec![(val - min) / (max - min) * (upper - lower) + lower]
                    }
                };

                match col.location {
                    Location::Input => newin.extend(newval),
                    Location::Target => newout.extend(newval),
                };
            }
            indata.push(newin);
            outdata.push(newout);
        }
        (indata, outdata)
    }

    /// returns the mean and standard deviation for the column
    /// Uses both train and test data to determine
    fn get_mean_sd(&self) -> HashMap<usize, (f32, f32)> {
        let mut mean_var: HashMap<usize, (f32, f32)> = HashMap::default();
        for col in self.columns.iter() {
            if let Adjustment::NormaliseMean = col.adjustment {
                let vals = self
                    .data
                    .iter()
                    .chain(&self.test_data)
                    .map(|line| line[col.index].parse::<f32>().unwrap_or_default())
                    .collect::<Vec<_>>();

                let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
                let sd = if vals.len() <= 1 {
                    0.
                } else {
                    (vals.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                        / (vals.len() as f32 - 1.0))
                        .sqrt()
                };

                mean_var.insert(col.index, (mean, sd));
            }
        }

        mean_var
    }
    /// returns values between lower and upper bounds
    /// Uses both train and test data to determine min/max
    fn get_min_max(&self) -> HashMap<usize, (f32, f32)> {
        let mut min_max: HashMap<usize, (f32, f32)> = HashMap::default();
        for col in self.columns.iter() {
            if let Adjustment::NormaliseMinMax(_, _) = col.adjustment {
                let vals = self
                    .data
                    .iter()
                    .chain(&self.test_data)
                    .map(|line| line[col.index].parse::<f32>().unwrap_or_default())
                    .collect::<Vec<_>>();

                let min: f32 = vals
                    .iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).expect("No NAN"))
                    .expect("Data is not empty");
                let max: f32 = vals
                    .iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).expect("No NAN"))
                    .expect("Data is not empty");

                min_max.insert(col.index, (min, max));
            }
        }

        min_max
    }

    /// Returns index, HashMap of String:correct f32 vec
    /// Uses both train and test data because there may be items in test data which are not in training data
    fn get_one_hot(&self) -> HashMap<usize, HashMap<String, Vec<f32>>> {
        let mut onehot: HashMap<usize, HashMap<String, Vec<f32>>> = HashMap::default();
        for col in self.columns.iter() {
            if let Adjustment::OneHot = col.adjustment {
                let uniq = self
                    .data
                    .iter()
                    .chain(&self.test_data)
                    .map(|line| line[col.index].to_owned())
                    .collect::<HashSet<String>>();
                let mut vals = Vec::from_iter(uniq);

                vals.sort_by_key(|a| a.to_lowercase());

                let mut hash = HashMap::new();
                for (i, str) in vals.iter().enumerate() {
                    let mut vec: Vec<f32> = std::iter::repeat(0.).take(vals.len()).collect();
                    vec[i] = 1.;
                    hash.insert(str.to_owned(), vec);
                }
                onehot.insert(col.index, hash);
            }
        }

        onehot
    }

    fn asserts(&self) {
        assert!(!self.data.is_empty(), "No data");
        assert!(
            self.columns
                .iter()
                .filter(|x| x.location == Location::Input)
                .count()
                > 0,
            "No input columns"
        );
        assert!(
            self.columns
                .iter()
                .filter(|x| x.location == Location::Target)
                .count()
                > 0,
            "No target columns"
        );

        //check every line is the same
        let first_len = self.data[0].len();
        for l in self.data.iter().chain(&self.test_data) {
            assert_eq!(
                first_len,
                l.len(),
                "Some columns do not have the same number of columns as first line {l:?}"
            )
        }

        let (mintrainlen, trainline) = self
            .data
            .iter()
            .map(|x| (x.len(), x))
            .min_by(|a, b| a.0.cmp(&b.0))
            .expect("Train data empty");

        let maxcolindex = self
            .columns
            .iter()
            .map(|x| x.index)
            .max()
            .unwrap_or(usize::MAX);

        assert!(maxcolindex < mintrainlen,"A line in the train data does not have enough columns ({trainline:?}) ({mintrainlen}) vs max column index {} (column {})",maxcolindex,maxcolindex+1);

        //test data may be empty, if not, we do some tests:
        if !self.test_data.is_empty() {
            let (mintestlen, testline) = self
                .test_data
                .iter()
                .map(|x| (x.len(), x))
                .min_by(|a, b| a.0.cmp(&b.0))
                .expect("Test data empty");

            assert!(maxcolindex < mintestlen,"A line in the test data does not have enough columns ({testline:?}) ({mintestlen}) vs max column index {} (column {})",maxcolindex,maxcolindex+1);
        }

        if !self.headers.is_empty() {
            assert_eq!(
                self.headers.len(),
                first_len,
                "Header count is not equal to line column count"
            );
        }
    }
}

#[derive(Debug)]
struct ColumnStats {
    pub mean_sd: HashMap<usize, (f32, f32)>,
    pub min_max: HashMap<usize, (f32, f32)>,
    pub one_hot: HashMap<usize, HashMap<String, Vec<f32>>>,
}

#[cfg(test)]
mod tests {
    use crate::{dataset::Adjustment, nn::NN};

    use super::Dataset;

    #[test]
    fn shuf() {
        fastrand::seed(1);
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        let set = Dataset::builder()
            .add_data(data)
            .add_input_columns(&[0, 1], Adjustment::F32)
            .add_target_columns(&[2], Adjustment::F32)
            .build();
        let (inp, tar) = set.get_data();
        //with this seed, the next shuffle should result in a different selection
        let (inp2, tar2) = set.get_data();
        assert_ne!(inp, inp2);
        assert_ne!(tar, tar2);
    }

    #[test]
    fn test_allocate_test() {
        fastrand::seed(3);
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        let set = Dataset::builder()
            .add_data(data)
            .add_input_columns(&[0, 1], Adjustment::F32)
            .add_target_columns(&[2], Adjustment::F32)
            .allocate_to_test_data(0.5)
            .build();
        assert_eq!(set.data.len(), 1);
        assert_eq!(set.test_data.len(), 1);
    }

    #[test]
    fn test_allocate_range() {
        fastrand::seed(3);
        let data = vec![
            vec!["1", "2", "3"],
            vec!["4", "5", "6"],
            vec!["7", "8", "9"],
        ];
        let set = Dataset::builder()
            .add_data(data)
            .add_input_columns(&[0, 1], Adjustment::F32)
            .add_target_columns(&[2], Adjustment::F32)
            .allocate_range_to_test_data(1..=1)
            .build();
        assert_eq!(set.data.len(), 2);
        assert_eq!(set.test_data.len(), 1);
        assert_eq!(
            set.test_data
                .iter()
                .map(|x| x.0.iter().map(|y| format!("{:0}", y)).collect())
                .collect::<Vec<Vec<String>>>(),
            &[["4", "5"]]
        );
    }

    #[test]
    fn test_nn() {
        fastrand::seed(3);
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        let set = Dataset::builder()
            .add_data(data)
            .add_input_columns(&[0, 1], Adjustment::F32)
            .add_target_columns(&[2], Adjustment::F32)
            .allocate_to_test_data(0.5)
            .build();

        let mut net = NN::new(&[2, 1]);
        let data = set.get_data();
        let test = set.get_test_data();

        net.fit(&data.0, &data.1);
        net.forward_error(&data.0[0], &data.1[0]);
        net.forward_error(&test.0[0], &test.1[0]);
    }
    #[test]
    fn test_get() {
        fastrand::seed(3);
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        let set = Dataset::builder()
            .add_data(data)
            .allocate_to_test_data(0.5)
            .add_input_columns(&[0, 1], Adjustment::F32)
            .add_target_columns(&[2], Adjustment::F32)
            .build();

        assert!(set.get_one().0.len() == 2);
        assert!(set.get_one_test().0.len() == 2);
        assert!(set.get_data().0[0].len() == 2);
        assert!(set.get_test_data().0[0].len() == 2);
    }

    #[test]
    #[should_panic]
    fn test_assert_nodata() {
        Dataset::builder().build();
    }

    #[test]
    #[should_panic]
    fn test_assert_noinputcolumns() {
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        Dataset::builder().add_data(data).build();
    }

    #[test]
    #[should_panic]
    fn test_assert_notargetcolumns() {
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        Dataset::builder()
            .add_data(data)
            .add_input_columns(&[0], Adjustment::F32)
            .build();
    }

    #[test]
    fn test_builds_with_basic() {
        let data = vec![vec!["1", "2", "3"], vec!["4", "5", "6"]];
        Dataset::builder()
            .add_data(data)
            .add_input_columns(&[0], Adjustment::F32)
            .add_target_columns(&[2], Adjustment::F32)
            .build();
    }
}
