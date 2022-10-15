///////////////////////////////////////////////////////////////////////////
/// PROGRAMACIÓN EN CUDA C/C++
/// Práctica:	BASICO 5 : Sincronización
/// Autor:		Gustavo Gutierrez Martin
/// Fecha:		Octubre 2022
///////////////////////////////////////////////////////////////////////////

/// dependencias ///
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

/// constantes ///
#define MB (1<<20) /// MiB = 2^20
#define PI 3.141593F /// numero PI a comparar

/// muestra por consola que no se ha encontrado un dispositivo CUDA
int getErrorDevice();
/// muestra los datos de los dispositivos CUDA encontrados
int getDataDevice(int deviceCount);
/// numero de CUDA cores
int getCudaCores(cudaDeviceProp deviceProperties);
/// muestra por pantalla las propiedades del dispositivo CUDA
int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties);
/// inicializa el array del host
/// solicita al usuario el número de elementos que se sumaran
int requestNumberOfTerms(int *numberOfItems, int maxThreadsPerBlock);
/// realiza la suma de los arrays en el device
__global__ void reduction(float *dev_datos, float *dev_suma);
/// transferimos los datos del device al host
int dataTransferToHost(float *hst_suma, float *dev_suma);
/// función que muestra por pantalla la salida del programa
int getAppOutput();

int main() {
    int deviceCount, maxThreadsPerBlock;
    int numberOfTerms = 0;
    float pi_calculate, relative_err, abs_err;
    float *hst_suma;
    float *dev_datos,*dev_suma;

    /// buscando dispositivos
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        /// mostramos el error si no se encuentra un dispositivo
        return getErrorDevice();
    } else {
        /// mostramos los datos de los dispositivos CUDA encontrados
        maxThreadsPerBlock = getDataDevice(deviceCount);
    }
    /// solicitamos al usuario la cantidad de elementos
    requestNumberOfTerms(&numberOfTerms, maxThreadsPerBlock);
    /// reserva del espacio de memoria en el host
    hst_suma = (float*)malloc(sizeof(float));
    /// reserva del espacio de memoria en el device
    cudaMalloc( (void**)&dev_datos, numberOfTerms * sizeof(float));
    cudaMalloc( (void**)&dev_suma, sizeof(float));
    /// imprimimos por pantalla los hilos lanzados
    printf("Lanzamiento de: 1 bloque y %d hilos \n", numberOfTerms);
    /// sumamos los items
    reduction<<< 1, numberOfTerms >>>(dev_datos, dev_suma);
    /// transferimos los datos del device al host
    dataTransferToHost(hst_suma,dev_suma);
    /// calculamos el valor de PI
    pi_calculate = sqrt(6 * hst_suma[0]);
    /// hallamos el error absoluto
    abs_err = pi_calculate - PI;
    /// hallamos el error relativo
    relative_err = (abs_err / PI) * 100;
    printf("> Valor de PI \t\t: %.6f \n", PI);
    printf("> Valor calculado \t: %.6f \n", pi_calculate);
    printf("> Error absoluto \t: %.6f \n", abs_err);
    printf("> Error relativo \t: %.6f%% \n", relative_err);
    /// función que muestra por pantalla la salida del programa
    getAppOutput();
    /// liberamos los recursos del device
    cudaFree(dev_datos);
    cudaFree(dev_suma);
    return 0;
}

int getErrorDevice() {
    printf("¡No se ha encontrado un dispositivo CUDA!\n");
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 1;
}

int getDataDevice(int deviceCount) {
    printf("Se han encontrado %d dispositivos CUDA:\n", deviceCount);
    int maxThreadsPerBlock = 0;
    for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
        ///obtenemos las propiedades del dispositivo CUDA
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, deviceID);
        getDeviceProperties(deviceID, getCudaCores(deviceProp), deviceProp);
        maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    }
    return maxThreadsPerBlock;
}

int getCudaCores(cudaDeviceProp deviceProperties) {
    int cudaCores = 0;
    int major = deviceProperties.major;
    if (major == 1) {
        /// TESLA
        cudaCores = 8;
    } else if (major == 2) {
        /// FERMI
        if (deviceProperties.minor == 0) {
            cudaCores = 32;
        } else {
            cudaCores = 48;
        }
    } else if (major == 3) {
        /// KEPLER
        cudaCores = 192;
    } else if (major == 5) {
        /// MAXWELL
        cudaCores = 128;
    } else if (major == 6 || major == 7 || major == 8) {
        /// PASCAL, VOLTA (7.0), TURING (7.5), AMPERE
        cudaCores = 64;
    } else {
        /// ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("¡Dispositivo desconocido!\n");
    }
    return cudaCores;
}

int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties) {
    int SM = cudaProperties.multiProcessorCount;
    printf("***************************************************\n");
    printf("DEVICE %d: %s\n", deviceId, cudaProperties.name);
    printf("***************************************************\n");
    printf("- Capacidad de Computo            \t: %d.%d\n", cudaProperties.major, cudaProperties.minor);
    printf("- No. de MultiProcesadores        \t: %d \n", SM);
    printf("- No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
    printf("- Memoria Global (total)          \t: %zu MiB\n", cudaProperties.totalGlobalMem / MB);
    printf("- No. maximo de Hilos (por bloque)\t: %d\n", cudaProperties.maxThreadsPerBlock);
    printf("***************************************************\n");
    return 0;
}

int requestNumberOfTerms(int *numberOfItems, int maxThreadsPerBlock) {
    int status = 0;
    while (status == 0) {
        printf("Introduce el numero de terminos (potencia de 2): \n");
        scanf_s("%d", numberOfItems);
        if (ceil(log2(*numberOfItems)) == floor(log2(*numberOfItems)) && *numberOfItems <= maxThreadsPerBlock ) {
            printf("El numero de elementos elegido es: %d \n", *numberOfItems);
            status = 1;
        } else {
            printf("El numero maximo de terminos no es potencia de 2 o supera el numero maximo de hilos por bloque \n");
        }
    }
    return 0;
}

__global__ void reduction(float *dev_datos, float *dev_suma) {
    /// KERNEL con 1 bloque de N hilos
    unsigned int N = blockDim.x;
    /// indice local de cada hilo
    unsigned int myID = threadIdx.x;
    /// rellenamos el vector de datos
    unsigned int term = myID + 1;
    dev_datos[myID] = (float)(1.0 / (term * term));
    /// sincronizamos para evitar riesgos de tipo RAW
    __syncthreads();
    /// ******************
    /// REDUCCION PARALELA
    /// ******************
    int salto = N / 2;
    /// realizamos log2(N) iteraciones
    while (salto > 0) {
        /// en cada paso solo trabajan la mitad de los hilos
        if (myID < salto) {
            dev_datos[myID] = dev_datos[myID] + dev_datos[myID + salto];
        }
        /// sincronizamos los hilos evitar riesgos de tipo RAW
        __syncthreads();
        salto = salto / 2;
    }
    /// ******************
    /// Solo el hilo no.'0' escribe el resultado final:
    /// evitamos los riesgos estructurales por el acceso a la memoria
    if (myID == 0) {
        *dev_suma = dev_datos[0];
    }
}

int dataTransferToHost(float *hst_suma, float *dev_suma) {
    /// transfiere datos de dev_vector2 a hst_vector2
    cudaMemcpy(hst_suma, dev_suma, sizeof(int), cudaMemcpyDeviceToHost);
    return 0;
}

int getAppOutput() {
    /// salida del programa
    time_t fecha;
    time(&fecha);
    printf("***************************************************\n");
    printf("Programa ejecutado el: %s", ctime(&fecha));
    printf("***************************************************\n");
    /// capturamos un INTRO para que no se cierre la consola de MSVS
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 0;
}


