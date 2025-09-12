function mean(arr) {
    if (arr.length === 0) {
        return 0;
    }
    const sum = arr.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    return sum / arr.length;
}

function variance(arr, usePopulation = false) {
    const n = arr.length;
    if (n <= 1) return 0;

    const meanValue = arr.reduce((a, b) => a + b) / n;
    const deviations = arr.map(x => x - meanValue);
    const sumOfSquaredDeviations = deviations.reduce((a, b) => a + b ** 2, 0);
    const divisor = usePopulation ? n : n - 1;

    return sumOfSquaredDeviations / divisor;
}

let n_bin = cb_obj.value;
let time = original.data.time;
let flux = original.data.flux;
let flux_err = original.data.flux_err;
let model = original.data.model;
let upper = original.data.upper;
let lower = original.data.lower;

let x_bins = [];
let y_bins = [];
let y_err_bins = [];
let model_bins = [];

for (let i = 0; i < time.length; i += n_bin) {
    const timeSlice = time.slice(i, i + n_bin);
    const fluxSlice = flux.slice(i, i + n_bin);
    const modelSlice = model.slice(i, i + n_bin);

    x_bins.push(mean(timeSlice));
    y_bins.push(mean(fluxSlice));
    model_bins.push(mean(modelSlice));
    y_err_bins.push(Math.sqrt(variance(fluxSlice)) / Math.sqrt(fluxSlice.length));
}

let y_err_upper = y_bins.map((value, index) => value + y_err_bins[index]);
let y_err_lower = y_bins.map((value, index) => value - y_err_bins[index]);

time = x_bins;
flux = y_bins;
flux_err = y_err_bins;
model = model_bins;
upper = y_err_upper;
lower = y_err_lower;

source.data = { time, flux, flux_err, model, upper, lower };