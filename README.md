- Schöne Print Ausgabe noch ergänzen
- Vielleicht Data direkt über einen Link aus Kaggle laden

## Matrix-Debugging

```bash
make DEBUG=1
gdb ./build/matrix_test
```

In GDB den Transpose-Operator anhalten und die Indizes sowie Faktoren prüfen:

```gdb
break 'operator*(Matrix<(Layout)1> const&, TransposeExpr<(Layout)1> const&)'
run
next
print i
print j
print k
print A(k, j)
print B(j, i)
print B.ref(i, j)
```

## Matrix-Benchmark

```bash
make benchmark
./build/matrix_benchmark
./build/matrix_benchmark 200
```

Der Benchmark misst drei Multiplikationspfade bei `784 x 64` mal `64 x 96`.
Die optionale Zahl bestimmt die Anzahl der Wiederholungen.
