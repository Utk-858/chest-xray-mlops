import http from 'k6/http';
import { check } from 'k6';

export const options = {
    vus: 25,
    duration: '30s',
};

const image = open('../data/test-image.jpeg', 'b');

export default function () {

    const payload = {
    file: http.file(
        image,
        "test-image.jpeg",
        "image/jpeg"
    )
};

    const res = http.post(
        'http://13.60.97.222:8000/api/v1/predict',
        payload
    );

    check(res, {
        'status is 200': (r) => r.status === 200,
    });
}