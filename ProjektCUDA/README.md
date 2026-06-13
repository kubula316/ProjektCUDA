# Projekt: Path Tracer CUDA Benchmark

# Autor: Jakub Siudek

# 

# Wymagania do zbudowania projektu:

# \- Zainstalowane CUDA Toolkit

# \- Biblioteki: GLFW3, GLEW, OpenGL (z OpenMP dla wersji CPU)

# \- Narzędzie budujące: CMake (wersja min. 3.18)

# 

# Instrukcja kompilacji (Linux / Windows CMake):

# 1\. Otwórz terminal/wiersz poleceń w folderze projektu.

# 2\. Wykonaj sekwencję komend:

# &#x20;  mkdir build

# &#x20;  cd build

# &#x20;  cmake ..

# &#x20;  cmake --build . --config Release

# 

# Sterowanie w aplikacji:

# \- Klawisze W, A, S, D + Myszka: Poruszanie się po scenie

# \- Klawisz \[SPACE]: Przełączanie w locie między GPU a CPU

# \- Klawisz \[B]: Uruchomienie automatycznego benchmarku (wyniki dopiszą się do pliku CSV)

# \- Klawisz \[ESC]: Wyjście z programu

