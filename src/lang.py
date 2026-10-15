"""Minimal UI translation support for k2p2viewer.

The English text itself is the lookup key, so untranslated strings fall
back to English automatically. Use tr('English text') for every UI string;
format placeholders are applied by the caller, e.g. tr('reading {0} sets').format(n).
"""

LANGUAGES = {'en': 'English', 'es': 'Español', 'pt': 'Português',
             'fr': 'Français', 'de': 'Deutsch'}

_current = 'en'


def set_language(code):
    global _current
    _current = code if code in LANGUAGES else 'en'


def get_language():
    return _current


def tr(text):
    if _current == 'en':
        return text
    return _TRANSLATIONS.get(_current, {}).get(text, text)


_ES = {
    # --- splash screen
    'Loading interface...':   'Cargando interfaz...',
    'Loading numpy...':       'Cargando numpy...',
    'Loading dataset...':     'Cargando conjunto de datos...',
    'Loading tools...':       'Cargando herramientas...',
    'Starting application...': 'Iniciando aplicación...',

    # --- main window / control bar
    'Reload':          'Recargar',
    'drop 0':          'descartar 0',
    'drop 1':          'descartar 1',
    'drop 2':          'descartar 2',
    'order':           'orden',
    'use sinc':        'usar sinc',
    'ignore cache':    'ignorar caché',
    'force scan all':  'forzar escaneo completo',
    'Language':        'Idioma',
    'Click on a run':  'Haga clic en una medición',
    'Click on + to add a balance with data directory':
        'Haga clic en + para añadir una balanza con su directorio de datos',
    'Welcome to k2viewer': 'Bienvenido al visor Kibb-g2',
    'Kibb-g2 Viewer':      'Visor Kibb-g2',
    'n/a':             'n/d',
    'Language changed': 'Idioma cambiado',
    'Restart k2p2viewer now to apply the new language?':
        '¿Reiniciar k2p2viewer ahora para aplicar el nuevo idioma?',

    # --- run table
    'run':      'medición',
    'title':    'título',
    'mass/mg':  'masa/mg',
    'unc/ug':   'inc./ug',

    # --- context menus
    'Rename':                  'Renombrar',
    'Delete':                  'Eliminar',
    'Change Directory...':     'Cambiar directorio...',
    'Copy table to clipboard': 'Copiar tabla al portapapeles',
    'Table copied to clipboard': 'Tabla copiada al portapapeles',

    # --- balance management dialogs
    'Invalid name':   'Nombre no válido',
    'Already exists': 'Ya existe',
    'Balance name may not contain [ or ] characters.':
        'El nombre de la balanza no puede contener los caracteres [ o ].',
    'Balance name cannot be empty.':
        'El nombre de la balanza no puede estar vacío.',
    'A balance named "{0}" already exists.':
        'Ya existe una balanza llamada "{0}".',
    'Confirm Delete': 'Confirmar eliminación',
    'Delete balance "{0}"?\n\nThis will remove it from the config and '
    'delete all its database entries. This cannot be undone.':
        '¿Eliminar la balanza "{0}"?\n\nSe eliminará de la configuración y '
        'se borrarán todas sus entradas de la base de datos. Esta acción no se puede deshacer.',
    'Select new data directory for "{0}"':
        'Seleccione el nuevo directorio de datos para "{0}"',
    'Select data directory for new balance':
        'Seleccione el directorio de datos para la nueva balanza',
    'Select data directory': 'Seleccione el directorio de datos',
    'Add Balance':    'Añadir balanza',
    'Balance name':   'Nombre de la balanza',
    'Balance setup':  'Configuración de la balanza',
    'Data path: {0}\n\nEnter a name for this balance\n(no [ or ] characters allowed):':
        'Ruta de datos: {0}\n\nIntroduzca un nombre para esta balanza\n(no se permiten los caracteres [ o ]):',
    'No balance configured yet.\n\nPlease select its data directory.':
        'Todavía no hay ninguna balanza configurada.\n\nSeleccione su directorio de datos.',
    'The configured data path does not exist:\n\n{0}\n\nPlease select the data directory.':
        'La ruta de datos configurada no existe:\n\n{0}\n\nSeleccione el directorio de datos.',
    'Data path not configured for {0}':
        'Ruta de datos no configurada para {0}',

    # --- busy / status messages
    'Wait until table is populated': 'Espere hasta que la tabla esté completa',
    'Wait until run is loaded':      'Espere hasta que la medición esté cargada',
    'Wait till last run is processed':
        'Espere hasta que se procese la última medición',
    'reading {0} sets':        'leyendo {0} conjuntos',
    'reading Environmentals':  'leyendo datos ambientales',
    'Loading from cache...':   'Cargando desde caché...',
    'reading of {0} done':     'lectura de {0} terminada',
    'Data available':          'Datos disponibles',

    # --- update dialog
    'Update available': 'Actualización disponible',
    'A new version is available: v{0}\nYou have: v{1}\n\nOpen the download page?':
        'Hay una nueva versión disponible: v{0}\nVersión instalada: v{1}\n\n¿Abrir la página de descarga?',

    # --- tabs
    'Mass':           'Masa',
    'Force':          'Fuerza',
    'Environmentals': 'Ambientales',
    'Velocity':       'Velocidad',
    'Profile':        'Perfil',
    'Uncertainty':    'Incertidumbre',
    'Report':         'Informe',
    'Allan':          'Allan',
    'Diagnostics':    'Diagnóstico',

    # --- tab controls
    'Forcemode':      'Modo fuerza',
    'show voltage':   'mostrar tensión',
    'Velocitymode':   'Modo velocidad',
    'ref. mass (mg):': 'masa ref. (mg):',
    'mass':           'masa',
    'Excl. outlier':  'Excl. valores atípicos',
    'show ppm':       'mostrar ppm',
    'Measured Mass:': 'Masa medida:',
    'Difference from reference:': 'Diferencia con la referencia:',
    'Item':           'Concepto',
    'rel. unc.':      'inc. rel.',
    'uncertainty':    'incertidumbre',
    'measured mass':  'masa medida',

    # --- mass results
    '± {0:4.1f} µg (Type A)': '± {0:4.1f} µg (Tipo A)',
    '± {0:4.1f} µg (Type B)': '± {0:4.1f} µg (Tipo B)',
    'PRTData.dat not found':           'PRTData.dat no encontrado',
    'PRTData.dat contains only zeros': 'PRTData.dat contiene solo ceros',
    'PRTData.dat could not be read':   'No se pudo leer PRTData.dat',
    'PRT data compromised':            'Datos PRT comprometidos',
    '⚠ {0} — using standard values\n(T=20 °C, P=999.9 hPa, rH=40 %)':
        '⚠ {0} — usando valores estándar\n(T=20 °C, P=999.9 hPa, HR=40 %)',
    '{0} PRT dropout(s) skipped': '{0} pérdida(s) de PRT omitida(s)',
    '±{0:.1f} µg overall (k={1})': '±{0:.1f} µg total (k={1})',
    '±{0:.1f} µg Type A':          '±{0:.1f} µg Tipo A',

    # --- plot axes
    'rel. humid (%)':     'hum. rel. (%)',
    'press. (hPa)':       'presión (hPa)',
    'temp (degC)':        'temp. (°C)',
    'air dens. (kg/m^3)': 'dens. aire (kg/m^3)',
    'rel. change /ppm':   'cambio rel. /ppm',
    'deviation  /ppm':    'desviación  /ppm',
    'deviation  /µg': 'desviación  /µg',
    'mass /mg':           'masa /mg',
    'Not enough points for Allan deviation':
        'No hay suficientes puntos para la desviación de Allan',
    'Allan deviation':    'Desviación de Allan',
    'Number of averaged measurements  m': 'Número de mediciones promediadas  m',
    'Allan deviation / µg': 'Desviación de Allan / µg',

    # --- uncertainty budget items
    'Resistance':        'Resistencia',
    'Voltage':           'Tensión',
    'Mass position':     'Posición de la masa',
    'Verticality':       'Verticalidad',
    'Type A':            'Tipo A',
    'Total':             'Total',
    'Balance mechanics': 'Mecánica de la balanza',

    # --- report tab
    'Serial Number':       'Número de serie',
    'Weight designation':  'Designación de la pesa',
    'True mass':           'Masa verdadera',
    'Assumed density':     'Densidad supuesta',
    'Conventional mass':   'Masa convencional',
    'Deviation from ref.': 'Desviación de la ref.',
    'Total uncertainty':   'Incertidumbre total',
    'Tolerance (Class 3)': 'Tolerancia (Clase 3)',
    'Temperature':         'Temperatura',
    'Barometric pressure': 'Presión barométrica',
    'Humidity':            'Humedad',
    ' (n/a)':              ' (n/d)',

    # --- Excel export
    'Designated Weight': 'Pesa designada',
    'True Mass':         'Masa verdadera',
    'Assumed Dens.':     'Dens. supuesta',
    'Conv. Mass':        'Masa conv.',
    'Dev. from Nom.':    'Desv. del nom.',
    'Total Unc.':        'Inc. total',
    'Tolerance':         'Tolerancia',
    'Temp.':             'Temp.',
    'Press.':            'Presión',
    'Humid.':            'Humedad',
}

_PT = {  # Brazilian Portuguese
    # --- splash screen
    'Loading interface...':   'Carregando interface...',
    'Loading numpy...':       'Carregando numpy...',
    'Loading dataset...':     'Carregando conjunto de dados...',
    'Loading tools...':       'Carregando ferramentas...',
    'Starting application...': 'Iniciando aplicativo...',

    # --- main window / control bar
    'Reload':          'Recarregar',
    'drop 0':          'descartar 0',
    'drop 1':          'descartar 1',
    'drop 2':          'descartar 2',
    'order':           'ordem',
    'use sinc':        'usar sinc',
    'ignore cache':    'ignorar cache',
    'force scan all':  'forçar varredura completa',
    'Language':        'Idioma',
    'Click on a run':  'Clique em uma medição',
    'Click on + to add a balance with data directory':
        'Clique em + para adicionar uma balança com seu diretório de dados',
    'Welcome to k2viewer': 'Bem-vindo ao visualizador Kibb-g2',
    'Kibb-g2 Viewer':      'Visualizador Kibb-g2',
    'n/a':             'n/d',
    'Language changed': 'Idioma alterado',
    'Restart k2p2viewer now to apply the new language?':
        'Reiniciar o k2p2viewer agora para aplicar o novo idioma?',

    # --- run table
    'run':      'medição',
    'title':    'título',
    'mass/mg':  'massa/mg',
    'unc/ug':   'inc./ug',

    # --- context menus
    'Rename':                  'Renomear',
    'Delete':                  'Excluir',
    'Change Directory...':     'Alterar diretório...',
    'Copy table to clipboard': 'Copiar tabela para a área de transferência',
    'Table copied to clipboard': 'Tabela copiada para a área de transferência',

    # --- balance management dialogs
    'Invalid name':   'Nome inválido',
    'Already exists': 'Já existe',
    'Balance name may not contain [ or ] characters.':
        'O nome da balança não pode conter os caracteres [ ou ].',
    'Balance name cannot be empty.':
        'O nome da balança não pode estar vazio.',
    'A balance named "{0}" already exists.':
        'Já existe uma balança chamada "{0}".',
    'Confirm Delete': 'Confirmar exclusão',
    'Delete balance "{0}"?\n\nThis will remove it from the config and '
    'delete all its database entries. This cannot be undone.':
        'Excluir a balança "{0}"?\n\nEla será removida da configuração e '
        'todas as suas entradas no banco de dados serão apagadas. Esta ação não pode ser desfeita.',
    'Select new data directory for "{0}"':
        'Selecione o novo diretório de dados para "{0}"',
    'Select data directory for new balance':
        'Selecione o diretório de dados para a nova balança',
    'Select data directory': 'Selecione o diretório de dados',
    'Add Balance':    'Adicionar balança',
    'Balance name':   'Nome da balança',
    'Balance setup':  'Configuração da balança',
    'Data path: {0}\n\nEnter a name for this balance\n(no [ or ] characters allowed):':
        'Caminho dos dados: {0}\n\nDigite um nome para esta balança\n(os caracteres [ e ] não são permitidos):',
    'No balance configured yet.\n\nPlease select its data directory.':
        'Nenhuma balança configurada ainda.\n\nSelecione o diretório de dados.',
    'The configured data path does not exist:\n\n{0}\n\nPlease select the data directory.':
        'O caminho de dados configurado não existe:\n\n{0}\n\nSelecione o diretório de dados.',
    'Data path not configured for {0}':
        'Caminho de dados não configurado para {0}',

    # --- busy / status messages
    'Wait until table is populated': 'Aguarde até que a tabela esteja completa',
    'Wait until run is loaded':      'Aguarde até que a medição seja carregada',
    'Wait till last run is processed':
        'Aguarde até que a última medição seja processada',
    'reading {0} sets':        'lendo {0} conjuntos',
    'reading Environmentals':  'lendo dados ambientais',
    'Loading from cache...':   'Carregando do cache...',
    'reading of {0} done':     'leitura de {0} concluída',
    'Data available':          'Dados disponíveis',

    # --- update dialog
    'Update available': 'Atualização disponível',
    'A new version is available: v{0}\nYou have: v{1}\n\nOpen the download page?':
        'Uma nova versão está disponível: v{0}\nVersão instalada: v{1}\n\nAbrir a página de download?',

    # --- tabs
    'Mass':           'Massa',
    'Force':          'Força',
    'Environmentals': 'Ambientais',
    'Velocity':       'Velocidade',
    'Profile':        'Perfil',
    'Uncertainty':    'Incerteza',
    'Report':         'Relatório',
    'Allan':          'Allan',
    'Diagnostics':    'Diagnóstico',

    # --- tab controls
    'Forcemode':      'Modo força',
    'show voltage':   'mostrar tensão',
    'Velocitymode':   'Modo velocidade',
    'ref. mass (mg):': 'massa ref. (mg):',
    'mass':           'massa',
    'Excl. outlier':  'Excl. valores discrepantes',
    'show ppm':       'mostrar ppm',
    'Measured Mass:': 'Massa medida:',
    'Difference from reference:': 'Diferença em relação à referência:',
    'Item':           'Item',
    'rel. unc.':      'inc. rel.',
    'uncertainty':    'incerteza',
    'measured mass':  'massa medida',

    # --- mass results
    '± {0:4.1f} µg (Type A)': '± {0:4.1f} µg (Tipo A)',
    '± {0:4.1f} µg (Type B)': '± {0:4.1f} µg (Tipo B)',
    'PRTData.dat not found':           'PRTData.dat não encontrado',
    'PRTData.dat contains only zeros': 'PRTData.dat contém apenas zeros',
    'PRTData.dat could not be read':   'Não foi possível ler PRTData.dat',
    'PRT data compromised':            'Dados PRT comprometidos',
    '⚠ {0} — using standard values\n(T=20 °C, P=999.9 hPa, rH=40 %)':
        '⚠ {0} — usando valores padrão\n(T=20 °C, P=999.9 hPa, UR=40 %)',
    '{0} PRT dropout(s) skipped': '{0} falha(s) de PRT ignorada(s)',
    '±{0:.1f} µg overall (k={1})': '±{0:.1f} µg total (k={1})',
    '±{0:.1f} µg Type A':          '±{0:.1f} µg Tipo A',

    # --- plot axes
    'rel. humid (%)':     'umid. rel. (%)',
    'press. (hPa)':       'pressão (hPa)',
    'temp (degC)':        'temp. (°C)',
    'air dens. (kg/m^3)': 'dens. do ar (kg/m^3)',
    'rel. change /ppm':   'variação rel. /ppm',
    'deviation  /ppm':    'desvio  /ppm',
    'deviation  /µg':     'desvio  /µg',
    'mass /mg':           'massa /mg',
    'Not enough points for Allan deviation':
        'Pontos insuficientes para o desvio de Allan',
    'Allan deviation':    'Desvio de Allan',
    'Number of averaged measurements  m': 'Número de medições promediadas  m',
    'Allan deviation / µg': 'Desvio de Allan / µg',

    # --- uncertainty budget items
    'Resistance':        'Resistência',
    'Voltage':           'Tensão',
    'Mass position':     'Posição da massa',
    'Verticality':       'Verticalidade',
    'Type A':            'Tipo A',
    'Total':             'Total',
    'Balance mechanics': 'Mecânica da balança',

    # --- report tab
    'Serial Number':       'Número de série',
    'Weight designation':  'Designação do peso',
    'True mass':           'Massa verdadeira',
    'Assumed density':     'Densidade assumida',
    'Conventional mass':   'Massa convencional',
    'Deviation from ref.': 'Desvio da ref.',
    'Total uncertainty':   'Incerteza total',
    'Tolerance (Class 3)': 'Tolerância (Classe 3)',
    'Temperature':         'Temperatura',
    'Barometric pressure': 'Pressão barométrica',
    'Humidity':            'Umidade',
    ' (n/a)':              ' (n/d)',

    # --- Excel export
    'Designated Weight': 'Peso designado',
    'True Mass':         'Massa verdadeira',
    'Assumed Dens.':     'Dens. assumida',
    'Conv. Mass':        'Massa conv.',
    'Dev. from Nom.':    'Desv. do nom.',
    'Total Unc.':        'Inc. total',
    'Tolerance':         'Tolerância',
    'Temp.':             'Temp.',
    'Press.':            'Pressão',
    'Humid.':            'Umidade',
}

_FR = {
    # --- splash screen
    'Loading interface...':   "Chargement de l'interface...",
    'Loading numpy...':       'Chargement de numpy...',
    'Loading dataset...':     'Chargement des données...',
    'Loading tools...':       'Chargement des outils...',
    'Starting application...': "Démarrage de l'application...",

    # --- main window / control bar
    'Reload':          'Recharger',
    'drop 0':          'ignorer 0',
    'drop 1':          'ignorer 1',
    'drop 2':          'ignorer 2',
    'order':           'ordre',
    'use sinc':        'utiliser sinc',
    'ignore cache':    'ignorer le cache',
    'force scan all':  'forcer l\'analyse complète',
    'Language':        'Langue',
    'Click on a run':  'Cliquez sur une mesure',
    'Click on + to add a balance with data directory':
        'Cliquez sur + pour ajouter une balance avec son répertoire de données',
    'Welcome to k2viewer': 'Bienvenue dans la visionneuse Kibb-g2',
    'Kibb-g2 Viewer':      'Visionneuse Kibb-g2',
    'n/a':             'n/d',
    'Language changed': 'Langue modifiée',
    'Restart k2p2viewer now to apply the new language?':
        'Redémarrer k2p2viewer maintenant pour appliquer la nouvelle langue ?',

    # --- run table
    'run':      'mesure',
    'title':    'titre',
    'mass/mg':  'masse/mg',
    'unc/ug':   'incert./ug',

    # --- context menus
    'Rename':                  'Renommer',
    'Delete':                  'Supprimer',
    'Change Directory...':     'Changer de répertoire...',
    'Copy table to clipboard': 'Copier le tableau dans le presse-papiers',
    'Table copied to clipboard': 'Tableau copié dans le presse-papiers',

    # --- balance management dialogs
    'Invalid name':   'Nom non valide',
    'Already exists': 'Existe déjà',
    'Balance name may not contain [ or ] characters.':
        'Le nom de la balance ne peut pas contenir les caractères [ ou ].',
    'Balance name cannot be empty.':
        'Le nom de la balance ne peut pas être vide.',
    'A balance named "{0}" already exists.':
        'Une balance nommée « {0} » existe déjà.',
    'Confirm Delete': 'Confirmer la suppression',
    'Delete balance "{0}"?\n\nThis will remove it from the config and '
    'delete all its database entries. This cannot be undone.':
        'Supprimer la balance « {0} » ?\n\nElle sera retirée de la configuration et '
        'toutes ses entrées de la base de données seront effacées. Cette action est irréversible.',
    'Select new data directory for "{0}"':
        'Sélectionnez le nouveau répertoire de données pour « {0} »',
    'Select data directory for new balance':
        'Sélectionnez le répertoire de données de la nouvelle balance',
    'Select data directory': 'Sélectionnez le répertoire de données',
    'Add Balance':    'Ajouter une balance',
    'Balance name':   'Nom de la balance',
    'Balance setup':  'Configuration de la balance',
    'Data path: {0}\n\nEnter a name for this balance\n(no [ or ] characters allowed):':
        'Chemin des données : {0}\n\nSaisissez un nom pour cette balance\n(caractères [ et ] interdits) :',
    'No balance configured yet.\n\nPlease select its data directory.':
        'Aucune balance n\'est encore configurée.\n\nVeuillez sélectionner son répertoire de données.',
    'The configured data path does not exist:\n\n{0}\n\nPlease select the data directory.':
        'Le chemin de données configuré n\'existe pas :\n\n{0}\n\nVeuillez sélectionner le répertoire de données.',
    'Data path not configured for {0}':
        'Chemin de données non configuré pour {0}',

    # --- busy / status messages
    'Wait until table is populated': 'Attendez que le tableau soit rempli',
    'Wait until run is loaded':      'Attendez que la mesure soit chargée',
    'Wait till last run is processed':
        'Attendez que la dernière mesure soit traitée',
    'reading {0} sets':        'lecture de {0} séries',
    'reading Environmentals':  'lecture des données environnementales',
    'Loading from cache...':   'Chargement depuis le cache...',
    'reading of {0} done':     'lecture de {0} terminée',
    'Data available':          'Données disponibles',

    # --- update dialog
    'Update available': 'Mise à jour disponible',
    'A new version is available: v{0}\nYou have: v{1}\n\nOpen the download page?':
        'Une nouvelle version est disponible : v{0}\nVersion installée : v{1}\n\nOuvrir la page de téléchargement ?',

    # --- tabs
    'Mass':           'Masse',
    'Force':          'Force',
    'Environmentals': 'Environnement',
    'Velocity':       'Vitesse',
    'Profile':        'Profil',
    'Uncertainty':    'Incertitude',
    'Report':         'Rapport',
    'Allan':          'Allan',
    'Diagnostics':    'Diagnostic',

    # --- tab controls
    'Forcemode':      'Mode force',
    'show voltage':   'afficher la tension',
    'Velocitymode':   'Mode vitesse',
    'ref. mass (mg):': 'masse réf. (mg) :',
    'mass':           'masse',
    'Excl. outlier':  'Excl. valeurs aberrantes',
    'show ppm':       'afficher ppm',
    'Measured Mass:': 'Masse mesurée :',
    'Difference from reference:': 'Écart par rapport à la référence :',
    'Item':           'Composante',
    'rel. unc.':      'incert. rel.',
    'uncertainty':    'incertitude',
    'measured mass':  'masse mesurée',

    # --- mass results
    '± {0:4.1f} µg (Type A)': '± {0:4.1f} µg (type A)',
    '± {0:4.1f} µg (Type B)': '± {0:4.1f} µg (type B)',
    'PRTData.dat not found':           'PRTData.dat introuvable',
    'PRTData.dat contains only zeros': 'PRTData.dat ne contient que des zéros',
    'PRTData.dat could not be read':   'Impossible de lire PRTData.dat',
    'PRT data compromised':            'Données PRT compromises',
    '⚠ {0} — using standard values\n(T=20 °C, P=999.9 hPa, rH=40 %)':
        '⚠ {0} — valeurs standard utilisées\n(T=20 °C, P=999.9 hPa, HR=40 %)',
    '{0} PRT dropout(s) skipped': '{0} perte(s) de PRT ignorée(s)',
    '±{0:.1f} µg overall (k={1})': '±{0:.1f} µg global (k={1})',
    '±{0:.1f} µg Type A':          '±{0:.1f} µg type A',

    # --- plot axes
    'rel. humid (%)':     'hum. rel. (%)',
    'press. (hPa)':       'pression (hPa)',
    'temp (degC)':        'temp. (°C)',
    'air dens. (kg/m^3)': "masse vol. de l'air (kg/m^3)",
    'rel. change /ppm':   'variation rel. /ppm',
    'deviation  /ppm':    'écart  /ppm',
    'deviation  /µg':     'écart  /µg',
    'mass /mg':           'masse /mg',
    'Not enough points for Allan deviation':
        "Pas assez de points pour l'écart-type d'Allan",
    'Allan deviation':    "Écart-type d'Allan",
    'Number of averaged measurements  m': 'Nombre de mesures moyennées  m',
    'Allan deviation / µg': "Écart-type d'Allan / µg",

    # --- uncertainty budget items
    'Resistance':        'Résistance',
    'Voltage':           'Tension',
    'Mass position':     'Position de la masse',
    'Verticality':       'Verticalité',
    'Type A':            'Type A',
    'Total':             'Total',
    'Balance mechanics': 'Mécanique de la balance',

    # --- report tab
    'Serial Number':       'Numéro de série',
    'Weight designation':  'Désignation du poids',
    'True mass':           'Masse vraie',
    'Assumed density':     'Masse volumique supposée',
    'Conventional mass':   'Masse conventionnelle',
    'Deviation from ref.': 'Écart à la réf.',
    'Total uncertainty':   'Incertitude totale',
    'Tolerance (Class 3)': 'Tolérance (classe 3)',
    'Temperature':         'Température',
    'Barometric pressure': 'Pression barométrique',
    'Humidity':            'Humidité',
    ' (n/a)':              ' (n/d)',

    # --- Excel export
    'Designated Weight': 'Poids désigné',
    'True Mass':         'Masse vraie',
    'Assumed Dens.':     'Masse vol. supposée',
    'Conv. Mass':        'Masse conv.',
    'Dev. from Nom.':    'Écart au nom.',
    'Total Unc.':        'Incert. totale',
    'Tolerance':         'Tolérance',
    'Temp.':             'Temp.',
    'Press.':            'Pression',
    'Humid.':            'Humidité',
}

_DE = {
    # --- splash screen
    'Loading interface...':   'Oberfläche wird geladen...',
    'Loading numpy...':       'numpy wird geladen...',
    'Loading dataset...':     'Datensatz wird geladen...',
    'Loading tools...':       'Werkzeuge werden geladen...',
    'Starting application...': 'Anwendung wird gestartet...',

    # --- main window / control bar
    'Reload':          'Neu laden',
    'drop 0':          '0 verwerfen',
    'drop 1':          '1 verwerfen',
    'drop 2':          '2 verwerfen',
    'order':           'Ordnung',
    'use sinc':        'sinc verwenden',
    'ignore cache':    'Cache ignorieren',
    'force scan all':  'alles neu einlesen',
    'Language':        'Sprache',
    'Click on a run':  'Messung anklicken',
    'Click on + to add a balance with data directory':
        'Auf + klicken, um eine Waage mit Datenverzeichnis hinzuzufügen',
    'Welcome to k2viewer': 'Willkommen beim Kibb-g2 Viewer',
    'Kibb-g2 Viewer':      'Kibb-g2 Viewer',
    'n/a':             'k. A.',
    'Language changed': 'Sprache geändert',
    'Restart k2p2viewer now to apply the new language?':
        'k2p2viewer jetzt neu starten, um die neue Sprache zu übernehmen?',

    # --- run table
    'run':      'Messung',
    'title':    'Titel',
    'mass/mg':  'Masse/mg',
    'unc/ug':   'Uns./ug',

    # --- context menus
    'Rename':                  'Umbenennen',
    'Delete':                  'Löschen',
    'Change Directory...':     'Verzeichnis ändern...',
    'Copy table to clipboard': 'Tabelle in die Zwischenablage kopieren',
    'Table copied to clipboard': 'Tabelle in die Zwischenablage kopiert',

    # --- balance management dialogs
    'Invalid name':   'Ungültiger Name',
    'Already exists': 'Bereits vorhanden',
    'Balance name may not contain [ or ] characters.':
        'Der Name der Waage darf die Zeichen [ und ] nicht enthalten.',
    'Balance name cannot be empty.':
        'Der Name der Waage darf nicht leer sein.',
    'A balance named "{0}" already exists.':
        'Eine Waage mit dem Namen „{0}“ ist bereits vorhanden.',
    'Confirm Delete': 'Löschen bestätigen',
    'Delete balance "{0}"?\n\nThis will remove it from the config and '
    'delete all its database entries. This cannot be undone.':
        'Waage „{0}“ löschen?\n\nSie wird aus der Konfiguration entfernt und '
        'alle zugehörigen Datenbankeinträge werden gelöscht. Dies kann nicht rückgängig gemacht werden.',
    'Select new data directory for "{0}"':
        'Neues Datenverzeichnis für „{0}“ auswählen',
    'Select data directory for new balance':
        'Datenverzeichnis für die neue Waage auswählen',
    'Select data directory': 'Datenverzeichnis auswählen',
    'Add Balance':    'Waage hinzufügen',
    'Balance name':   'Name der Waage',
    'Balance setup':  'Waage einrichten',
    'Data path: {0}\n\nEnter a name for this balance\n(no [ or ] characters allowed):':
        'Datenpfad: {0}\n\nNamen für diese Waage eingeben\n(die Zeichen [ und ] sind nicht erlaubt):',
    'No balance configured yet.\n\nPlease select its data directory.':
        'Noch keine Waage konfiguriert.\n\nBitte das Datenverzeichnis auswählen.',
    'The configured data path does not exist:\n\n{0}\n\nPlease select the data directory.':
        'Der konfigurierte Datenpfad existiert nicht:\n\n{0}\n\nBitte das Datenverzeichnis auswählen.',
    'Data path not configured for {0}':
        'Kein Datenpfad für {0} konfiguriert',

    # --- busy / status messages
    'Wait until table is populated': 'Bitte warten, bis die Tabelle gefüllt ist',
    'Wait until run is loaded':      'Bitte warten, bis die Messung geladen ist',
    'Wait till last run is processed':
        'Bitte warten, bis die letzte Messung verarbeitet ist',
    'reading {0} sets':        'lese {0} Sätze',
    'reading Environmentals':  'lese Umgebungsdaten',
    'Loading from cache...':   'Lade aus dem Cache...',
    'reading of {0} done':     '{0} fertig eingelesen',
    'Data available':          'Daten verfügbar',

    # --- update dialog
    'Update available': 'Update verfügbar',
    'A new version is available: v{0}\nYou have: v{1}\n\nOpen the download page?':
        'Eine neue Version ist verfügbar: v{0}\nInstalliert: v{1}\n\nDownload-Seite öffnen?',

    # --- tabs
    'Mass':           'Masse',
    'Force':          'Kraft',
    'Environmentals': 'Umgebung',
    'Velocity':       'Geschwindigkeit',
    'Profile':        'Profil',
    'Uncertainty':    'Unsicherheit',
    'Report':         'Bericht',
    'Allan':          'Allan',
    'Diagnostics':    'Diagnose',

    # --- tab controls
    'Forcemode':      'Kraftmodus',
    'show voltage':   'Spannung anzeigen',
    'Velocitymode':   'Geschwindigkeitsmodus',
    'ref. mass (mg):': 'Ref.-Masse (mg):',
    'mass':           'Masse',
    'Excl. outlier':  'Ausreißer ausschl.',
    'show ppm':       'ppm anzeigen',
    'Measured Mass:': 'Gemessene Masse:',
    'Difference from reference:': 'Abweichung von der Referenz:',
    'Item':           'Beitrag',
    'rel. unc.':      'rel. Uns.',
    'uncertainty':    'Unsicherheit',
    'measured mass':  'gemessene Masse',

    # --- mass results
    '± {0:4.1f} µg (Type A)': '± {0:4.1f} µg (Typ A)',
    '± {0:4.1f} µg (Type B)': '± {0:4.1f} µg (Typ B)',
    'PRTData.dat not found':           'PRTData.dat nicht gefunden',
    'PRTData.dat contains only zeros': 'PRTData.dat enthält nur Nullen',
    'PRTData.dat could not be read':   'PRTData.dat konnte nicht gelesen werden',
    'PRT data compromised':            'PRT-Daten fehlerhaft',
    '⚠ {0} — using standard values\n(T=20 °C, P=999.9 hPa, rH=40 %)':
        '⚠ {0} — Standardwerte werden verwendet\n(T=20 °C, P=999.9 hPa, rF=40 %)',
    '{0} PRT dropout(s) skipped': '{0} PRT-Aussetzer übersprungen',
    '±{0:.1f} µg overall (k={1})': '±{0:.1f} µg gesamt (k={1})',
    '±{0:.1f} µg Type A':          '±{0:.1f} µg Typ A',

    # --- plot axes
    'rel. humid (%)':     'rel. Feuchte (%)',
    'press. (hPa)':       'Druck (hPa)',
    'temp (degC)':        'Temp. (°C)',
    'air dens. (kg/m^3)': 'Luftdichte (kg/m^3)',
    'rel. change /ppm':   'rel. Änderung /ppm',
    'deviation  /ppm':    'Abweichung  /ppm',
    'deviation  /µg':     'Abweichung  /µg',
    'mass /mg':           'Masse /mg',
    'Not enough points for Allan deviation':
        'Zu wenige Punkte für die Allan-Abweichung',
    'Allan deviation':    'Allan-Abweichung',
    'Number of averaged measurements  m': 'Anzahl gemittelter Messungen  m',
    'Allan deviation / µg': 'Allan-Abweichung / µg',

    # --- uncertainty budget items
    'Resistance':        'Widerstand',
    'Voltage':           'Spannung',
    'Mass position':     'Massenposition',
    'Verticality':       'Vertikalität',
    'Type A':            'Typ A',
    'Total':             'Gesamt',
    'Balance mechanics': 'Waagenmechanik',

    # --- report tab
    'Serial Number':       'Seriennummer',
    'Weight designation':  'Gewichtsbezeichnung',
    'True mass':           'Wahre Masse',
    'Assumed density':     'Angenommene Dichte',
    'Conventional mass':   'Konventioneller Wägewert',
    'Deviation from ref.': 'Abweichung von Ref.',
    'Total uncertainty':   'Gesamtunsicherheit',
    'Tolerance (Class 3)': 'Toleranz (Klasse 3)',
    'Temperature':         'Temperatur',
    'Barometric pressure': 'Luftdruck',
    'Humidity':            'Luftfeuchte',
    ' (n/a)':              ' (k. A.)',

    # --- Excel export
    'Designated Weight': 'Gewichtsbezeichnung',
    'True Mass':         'Wahre Masse',
    'Assumed Dens.':     'Angen. Dichte',
    'Conv. Mass':        'Konv. Wägewert',
    'Dev. from Nom.':    'Abw. vom Nennwert',
    'Total Unc.':        'Gesamtuns.',
    'Tolerance':         'Toleranz',
    'Temp.':             'Temp.',
    'Press.':            'Druck',
    'Humid.':            'Feuchte',
}

_TRANSLATIONS = {'es': _ES, 'pt': _PT, 'fr': _FR, 'de': _DE}
