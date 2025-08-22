import csv
import os

def read_csv_file(filename):
    """Leer el archivo CSV y devolver los datos"""
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def extract_group(image_name):
    """Extraer el grupo de la imagen (0, 1, 2)"""
    if image_name.endswith('0p.tif'):
        return '0'
    elif image_name.endswith('1p.tif'):
        return '1'
    elif image_name.endswith('2p.tif'):
        return '2'
    return 'Otro'

def format_value(value, decimal_places=4):
    """Formatear valores numéricos"""
    try:
        return f"{float(value):.{decimal_places}f}"
    except:
        return str(value)

def print_legend():
    """Imprimir leyenda con el significado de las siglas"""
    print("\n" + "="*80)
    print("LEYENDA - SIGNIFICADO DE LAS SIGLAS")
    print("="*80)
    print("VD  : Vein Density - Densidad de venas (proporción venas/área total)")
    print("VT  : Vein Thickness - Grosor promedio de las venas (píxeles)")
    print("AS  : Areole Size - Tamaño promedio de las aréolas (píxeles²)")
    print("NA  : Number of Areoles - Número de aréolas detectadas")
    print("BA  : Branching Angle - Ángulo promedio de ramificación (grados)")
    print("VVD : Vein-to-Vein Distance - Distancia promedio entre venas (píxeles)")
    print("="*80)

def create_comparison_table(data):
    """Crear una tabla comparativa organizada"""
    
    # Métricas a mostrar
    metrics = ['Vein Density (VD)', 'Vein Thickness (VT)', 'Areole Size (AS)', 
               'Number of Areoles (NA)', 'Branching Angle (BA)', 'Vein-to-Vein Distance (VVD)']
    
    # Organizar datos por grupo
    groups = {'0': [], '1': [], '2': [], 'Otro': []}
    for row in data:
        group = extract_group(row['image_name'])
        row['group'] = group
        groups[group].append(row)
    
    # Crear tabla comparativa
    print("TABLA COMPARATIVA DE MÉTRICAS POR IMAGEN")
    print("=" * 120)
    
    # Imprimir por grupos
    for group in ['0', '1', '2']:
        print(f"\n\nGRUPO {group}:")
        print("-" * 120)
        print(f"{'Imagen':<20} {'VD':<12} {'VT':<12} {'AS':<15} {'NA':<8} {'BA':<12} {'VVD':<12} {'Tamaño':<10}")
        print("-" * 120)
        
        for row in sorted(groups[group], key=lambda x: x['image_name']):
            vd = format_value(row['Vein Density (VD)'], 6)
            vt = format_value(row['Vein Thickness (VT)'], 4)
            as_val = format_value(row['Areole Size (AS)'], 2)
            na = format_value(row['Number of Areoles (NA)'], 0)
            ba = format_value(row['Branching Angle (BA)'], 2)
            vvd = format_value(row['Vein-to-Vein Distance (VVD)'], 2)
            
            print(f"{row['image_name']:<20} {vd:<12} {vt:<12} {as_val:<15} {na:<8} {ba:<12} {vvd:<12} {row['image_size']:<10}")

def create_individual_reports(data):
    """Crear reportes individuales para cada imagen"""
    
    metrics = ['Vein Density (VD)', 'Vein Thickness (VT)', 'Areole Size (AS)', 
               'Number of Areoles (NA)', 'Branching Angle (BA)', 'Vein-to-Vein Distance (VVD)']
    
    print("\n\nREPORTES INDIVIDUALES POR IMAGEN")
    print("=" * 80)
    
    for row in sorted(data, key=lambda x: x['image_name']):
        group = extract_group(row['image_name'])
        print(f"\nIMAGEN: {row['image_name']} (Grupo {group})")
        print(f"Tamaño: {row['image_size']}")
        print(f"Umbral: {row['threshold']}")
        print("-" * 60)
        
        for metric in metrics:
            value = format_value(row[metric], 6 if 'Density' in metric else 4)
            print(f"{metric:<30}: {value}")
        
        print("-" * 60)

def create_summary_by_group(data):
    """Crear resumen estadístico por grupo"""
    
    metrics = ['Vein Density (VD)', 'Vein Thickness (VT)', 'Areole Size (AS)', 
               'Number of Areoles (NA)', 'Branching Angle (BA)', 'Vein-to-Vein Distance (VVD)']
    
    groups = {'0': [], '1': [], '2': []}
    
    for row in data:
        group = extract_group(row['image_name'])
        if group in groups:
            groups[group].append(row)
    
    print("\n\nRESUMEN ESTADÍSTICO POR GRUPO")
    print("=" * 100)
    
    for group in ['0', '1', '2']:
        print(f"\nGRUPO {group} (n={len(groups[group])} imágenes):")
        print("-" * 100)
        
        for metric in metrics:
            values = [float(row[metric]) for row in groups[group] if row[metric] != '']
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"{metric:<30}: Promedio={avg:.6f}  Min={min_val:.6f}  Max={max_val:.6f}")

def create_detailed_legend():
    """Crear una leyenda detallada con explicaciones completas"""
    print("\n" + "="*100)
    print("LEYENDA DETALLADA - EXPLICACIÓN DE LAS MÉTRICAS")
    print("="*100)
    print("VD  - Vein Density (Densidad de venas):")
    print("     Proporción del área ocupada por venas respecto al área total de la imagen/pétalo")
    print("     Rango: 0-1 (0 = sin venas, 1 = completamente venoso)")
    print("     Unidad: Adimensional")
    print()
    print("VT  - Vein Thickness (Grosor de venas):")
    print("     Grosor promedio de las venas en la imagen")
    print("     Unidad: Píxeles (depende de la resolución de la imagen)")
    print()
    print("AS  - Areole Size (Tamaño de aréolas):")
    print("     Tamaño promedio de las áreas cerradas entre las venas")
    print("     Unidad: Píxeles² (depende de la resolución)")
    print()
    print("NA  - Number of Areoles (Número de aréolas):")
    print("     Cantidad total de áreas cerradas detectadas entre las venas")
    print("     Unidad: Conteo")
    print()
    print("BA  - Branching Angle (Ángulo de ramificación):")
    print("     Ángulo promedio en los puntos donde las venas se ramifican")
    print("     Unidad: Grados (°)")
    print()
    print("VVD - Vein-to-Vein Distance (Distancia entre venas):")
    print("     Distancia promedio entre venas adyacentes")
    print("     Unidad: Píxeles (depende de la resolución)")
    print("="*100)

def export_to_text_file(data, filename="analisis_individual.txt"):
    """Exportar todos los análisis a un archivo de texto"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Redirigir la salida al archivo
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        # Imprimir leyenda al inicio
        create_detailed_legend()
        
        # Ejecutar las funciones de análisis
        create_comparison_table(data)
        create_individual_reports(data)
        create_summary_by_group(data)
        
        # Imprimir leyenda al final también
        print("\n\n" + "="*100)
        print("LEYENDA RESUMEN")
        print("="*100)
        print("VD  : Densidad de venas (0-1)")
        print("VT  : Grosor de venas (px)")
        print("AS  : Tamaño de aréolas (px²)")
        print("NA  : Número de aréolas")
        print("BA  : Ángulo de ramificación (°)")
        print("VVD : Distancia entre venas (px)")
        print("="*100)
        
        # Restaurar stdout
        sys.stdout = original_stdout
    
    print(f"\nAnálisis completo exportado a: {filename}")

def main():
    # Leer datos
    data = read_csv_file('C:\\Users\\javir\\Documents\\TFG\\petalos\\results\\metrics_summary.csv')
    
    # Imprimir leyenda al inicio
    print_legend()
    
    # 1. Tabla comparativa
    create_comparison_table(data)
    
    # 2. Reportes individuales
    create_individual_reports(data)
    
    # 3. Resumen por grupo
    create_summary_by_group(data)
    
    # 4. Exportar a archivo (con leyenda incluida)
    export_to_text_file(data)
    
    # Imprimir leyenda resumen al final
    print("\n" + "="*80)
    print("LEYENDA RESUMEN")
    print("="*80)
    print("VD  : Densidad de venas")
    print("VT  : Grosor de venas (px)")
    print("AS  : Tamaño de aréolas (px²)")
    print("NA  : Número de aréolas")
    print("BA  : Ángulo de ramificación (°)")
    print("VVD : Distancia entre venas (px)")
    print("="*80)
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("Se han generado:")
    print("1. Tabla comparativa organizada por grupos")
    print("2. Reportes individuales para cada imagen")
    print("3. Resumen estadístico por grupo")
    print("4. Archivo de texto con todos los resultados + leyenda")
    print("="*80)

if __name__ == "__main__":
    main()