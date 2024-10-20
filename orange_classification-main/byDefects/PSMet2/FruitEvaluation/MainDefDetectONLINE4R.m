% ########################################################################
% Project AUTOMATIC CLASSIFICATION OF ORANGES BY SIZE AND DEFECTS USING 
% COMPUTER VISION TECHNIQUES 2018
% juancarlosmiranda81@gmail.com
% ########################################################################
% Se generan datos obtenidos luego de aplicar un método de segmentación y
% un clasificador de defectos previamente entrenado. Al final se obtiene un
% listado con las clasificaciones de lo detectado.



%% Ajuste de parámetros iniciales
clc; clear all; close all;
 
 %% Definicion de estructura de directorios 
%HOME=strcat(pwd,'/');
HOME=fullfile('C:','Users','Usuari','development','orange_classification');
pathPrincipal=fullfile(HOME,'OrangeResults','byDefects','PSMet2','FruitEvaluation'); %
pathEntradaImagenesTest=fullfile(HOME,'OrangeResults','inputTest');
pathConfiguracion=fullfile(pathPrincipal,'conf');
pathAplicacion=fullfile(pathPrincipal,'tmpToLearn'); % TODO: VEr porque este está anidado en el path del modulo
pathAplicacionSiluetas=fullfile(pathAplicacion,'sFrutas');
pathResultados=fullfile(pathPrincipal,'output');%se guardan los resultados

nombreImagenP='nombreImagenP';

%% Nombres de archivos de configuracion
% trabajan con métodos para equivalencia con las 4 vistas
 
%%
archivoConfiguracion=fullfile(pathConfiguracion,'20170916configuracion.xml'); %Para coordenadas iniciales en tratamiento de imagenes
archivoCalibracion=fullfile(pathConfiguracion,'20170916calibracion.xml'); %para indicar al usuario en la parte final la calibracion
  
 %% Definicion de los cuadros, según numeración 
Fila1=readConfiguration('Fila1', archivoConfiguracion);
FilaAbajo=readConfiguration('FilaAbajo', archivoConfiguracion);

%Cuadro 1 abajo
Cuadro1_lineaGuiaInicialFila=readConfiguration('Cuadro1_lineaGuiaInicialFila', archivoConfiguracion);
Cuadro1_lineaGuiaInicialColumna=readConfiguration('Cuadro1_lineaGuiaInicialColumna', archivoConfiguracion);
Cuadro1_espacioFila=readConfiguration('Cuadro1_espacioFila', archivoConfiguracion);
Cuadro1_espacioColumna=readConfiguration('Cuadro1_espacioColumna', archivoConfiguracion);

%Cuadro 2 izquierda
Cuadro2_lineaGuiaInicialFila=readConfiguration('Cuadro2_lineaGuiaInicialFila', archivoConfiguracion);
Cuadro2_lineaGuiaInicialColumna=readConfiguration('Cuadro2_lineaGuiaInicialColumna', archivoConfiguracion);
Cuadro2_espacioFila=readConfiguration('Cuadro2_espacioFila', archivoConfiguracion);
Cuadro2_espacioColumna=readConfiguration('Cuadro2_espacioColumna', archivoConfiguracion);

%Cuadro 3 centro
Cuadro3_lineaGuiaInicialFila=readConfiguration('Cuadro3_lineaGuiaInicialFila', archivoConfiguracion);
Cuadro3_lineaGuiaInicialColumna=readConfiguration('Cuadro3_lineaGuiaInicialColumna', archivoConfiguracion);
Cuadro3_espacioFila=readConfiguration('Cuadro3_espacioFila', archivoConfiguracion);
Cuadro3_espacioColumna=readConfiguration('Cuadro3_espacioColumna', archivoConfiguracion);

%Cuadro 4 derecha
Cuadro4_lineaGuiaInicialFila=readConfiguration('Cuadro4_lineaGuiaInicialFila', archivoConfiguracion);
Cuadro4_lineaGuiaInicialColumna=readConfiguration('Cuadro4_lineaGuiaInicialColumna', archivoConfiguracion);
Cuadro4_espacioFila=readConfiguration('Cuadro4_espacioFila', archivoConfiguracion);
Cuadro4_espacioColumna=readConfiguration('Cuadro4_espacioColumna', archivoConfiguracion);

%%carga en memoria para que sea mas rapido
ArrayCuadros=[Cuadro1_lineaGuiaInicialColumna, Cuadro1_lineaGuiaInicialFila, Cuadro1_espacioColumna, Cuadro1_espacioFila;
Cuadro2_lineaGuiaInicialColumna, Cuadro2_lineaGuiaInicialFila, Cuadro2_espacioColumna, Cuadro2_espacioFila;
Cuadro3_lineaGuiaInicialColumna, Cuadro3_lineaGuiaInicialFila, Cuadro3_espacioColumna, Cuadro3_espacioFila;
Cuadro4_lineaGuiaInicialColumna, Cuadro4_lineaGuiaInicialFila, Cuadro4_espacioColumna, Cuadro4_espacioFila;
0,0,0,0
];

%% CONFIGURACIONES DE PROCESAMIENTO DE IMAGENES
areaObjetosRemoverBR=5000; % para siluetas y detección de objetos. Tamaño para realizar granulometria
% configuracion de umbrales
canalLMin = 0.0; canalLMax = 96.653; canalAMin = -23.548; canalAMax = 16.303; canalBMin = -28.235; canalBMax = -1.169; %parametros de umbralizacion de fondo


%% CONFIGURACIONES PARA DETECCION DE DEFECTOS
tamanoManchas=1000; %se utiliza para extracción de contornos. Los contornos se encuentran arriba de 1000 pixeles
archivoVectorDef=fullfile(pathResultados,'aCandidatos.csv'); %archivo de salida candidatos a defectos

% ----- FIN Definicion de topes
%% Remover archivos antiguos, borrar archivos antiguos
fprintf('LIMPIANDO IMAGENES ANTIGUAS \n');
removeFiles(archivoVectorDef);
removeFiles(fullfile(pathAplicacion,'sFrutas','*.jpg'));
removeFiles(fullfile(pathAplicacion,'sDefectos','*.jpg'));
removeFiles(fullfile(pathAplicacion,'roi','*.jpg'));
removeFiles(fullfile(pathAplicacion,'removido','*.jpg'));
removeFiles(fullfile(pathAplicacion,'deteccion','*.jpg'));
removeFiles(fullfile(pathAplicacion,'defectos','*.jpg'));
removeFiles(fullfile(pathAplicacion,'contornos','*.jpg'));
removeFiles(fullfile(pathAplicacion,'cDefectos','*.jpg'));
removeFiles(fullfile(pathAplicacion,'br','*.jpg'));


%% --------------------------------------------------------------------
%carga del listado de nombres
listado=dir(fullfile(pathEntradaImagenesTest,'*.jpg'));

%% lectura en forma de bach del directorio de la cámara
for n=1:size(listado)
    fprintf('Extrayendo características para entrenamiento-> %s \n',listado(n).name);    
    nombreImagenP=listado(n).name;    

    ProcessImgSoft(pathEntradaImagenesTest, pathAplicacion, nombreImagenP, ArrayCuadros, areaObjetosRemoverBR, canalLMin, canalLMax, canalAMin, canalAMax, canalBMin, canalBMax )
    ExtractDefDetectImgSoft(pathEntradaImagenesTest, pathAplicacion, nombreImagenP, archivoVectorDef, tamanoManchas)
%    if n==1
%        break;
%    end %if n==11
end %

%total=size(listado);

fprintf('\n -------------------------------- \n');
fprintf('Se procesaron un total de %i archivos \n',n);
fprintf('Verificar los resultados del análisis en %s \n', archivoVectorDef)
fprintf('\n -------------------------------- \n');
