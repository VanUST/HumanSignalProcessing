#include <stdio.h>
#include <stdlib.h>

//параметры фильтра. менять можно q,r,p,x но по факту влияет только q,r; если интересно как работает рекомендую статью из хабра, правда там немного //другая терминология
typedef struct {
    float q;  // дисперсия реальных показаний
    float r;  // зашумленность измерений измерений
    float x;  // текущая оценка значения
    float p;  // дисперсия оценки значения
    float k;  // калман гейн (коэф внутренней логики)
} KalmanFilter;

void kalman_filter_init(KalmanFilter *kf, float process_noise, float measurement_noise, float estimated_error, float initial_value) {
    kf->q = process_noise;
    kf->r = measurement_noise;
    kf->p = estimated_error;
    kf->x = initial_value;
    kf->k = 0;
}

float kalman_filter_process(KalmanFilter *kf, float measurement) {
    // апдейтим статистику дисперсии
    kf->p = kf->p + kf->q;

    // апдейтим оценку измерения
    kf->k = kf->p / (kf->p + kf->r);
    kf->x = kf->x + kf->k * (measurement - kf->x);
    kf->p = (1 - kf->k) * kf->p;

    return kf->x;
}

int main() {

    unsigned short measurements[] = {512, 530, 480, 510, 495, 505, 490, 500, 515, 505};
    int num_measurements = sizeof(measurements) / sizeof(measurements[0]);

    // инициализируем фильтр параметрами
    KalmanFilter kf;
    kalman_filter_init(&kf, 1e-5, 10., 1.0, measurements[0]);

    // пример работы
    printf("Measurement\tKalman Filter Output\n");
    for (int i = 0; i < num_measurements; i++) {
        float filtered_value = kalman_filter_process(&kf, measurements[i]);
        printf("%d\t\t%.2f\n", measurements[i], filtered_value);
    }

    return 0;
}