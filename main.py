import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def carregar_e_limpar_transacoes(caminho_csv: str) -> pd.DataFrame:
    """
    1. Ingestão, Performance e Limpeza:
    - Carrega transacoes.csv com encoding 'latin1'.
    - Imputa NaN na coluna 'valor' com a mediana por estado.
    - Adiciona coluna estática 'plataforma' com valor 'Mobile'.
    """
    # Leitura garantindo encoding
    df = pd.read_csv(caminho_csv, encoding='latin1')
    
    # Imputação da mediana do 'valor' por 'estado_cliente'
    df['valor'] = df.groupby('estado_cliente')['valor'].transform(
        lambda grp: grp.fillna(grp.median())
    )
    
    # Adicionando coluna estática
    df['plataforma'] = 'Mobile'
    
    return df


def pipeline_engenharia_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    2. Engenharia de Dados & Alinhamento Temporal:
    - Converte data_transacao para datetime64[ns] e localiza fuso 'America/Sao_Paulo'.
    - Extrai dia da semana e mês.
    - Remove duplicatas mantendo a primeira ocorrência.
    """
    df = df.copy()
    
    # Conversão e fuso horário
    df['data_transacao'] = pd.to_datetime(df['data_transacao'])
    df['data_transacao'] = df['data_transacao'].dt.tz_localize('America/Sao_Paulo')
    
    # Acessores .dt para dia da semana e mês
    df['dia_semana'] = df['data_transacao'].dt.day_name()
    df['mes'] = df['data_transacao'].dt.month
    
    # Descarte de duplicatas
    df = df.drop_duplicates(keep='first')
    
    return df


def filtrar_transacoes_setembro(df: pd.DataFrame) -> pd.DataFrame:
    """
    3. Operações Vetorizadas e Filtros Bitwise:
    Filtra transações de Setembro que atendem:
    (estado == 'SP' OU estado == 'RJ') E (valor > 5000.00)
    """
    cond_mes = (df['mes'] == 9)
    cond_estado = (df['estado_cliente'] == 'SP') | (df['estado_cliente'] == 'RJ')
    cond_valor = (df['valor'] > 5000.00)
    
    df_filtrado = df[cond_mes & cond_estado & cond_valor].copy()
    return df_filtrado


def cruzar_dados_e_pivocar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    4. Cruzamento de Dados & Agregação:
    - Atribui risco via .map().
    - Cria pivot table de soma por mês x nivel_risco com subtotais.
    """
    df = df.copy()
    
    risco_dict = {
        'C100': 'Baixo',
        'C101': 'Alto',
        'C102': 'Médio',
        'C103': 'Alto',
        'C104': 'Baixo'
    }
    
    df['nivel_risco'] = df['id_cliente'].map(risco_dict).fillna('Indefinido')
    
    tabela_dinamica = pd.pivot_table(
        df,
        values='valor',
        index='mes',
        columns='nivel_risco',
        aggfunc='sum',
        margins=True,
        margins_name='Total'
    )
    
    return df, tabela_dinamica


def detectar_outliers_zscore(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    5. Detecção Estatística de Outliers (Z-Score):
    Função vetorizada sem laços for para calcular Z-Score do valor por estado.
    Filtra onde Z > 2.5.
    """
    df = df.copy()
    
    mean_by_state = df.groupby('estado_cliente')['valor'].transform('mean')
    std_by_state = df.groupby('estado_cliente')['valor'].transform('std')
    
    # Cálculo vetorizado do Z-Score: (X - mu) / sigma
    df['z_score'] = (df['valor'] - mean_by_state) / std_by_state
    
    anomalias = df[df['z_score'] > threshold].copy()
    return anomalias


def gerar_relatorio_visual(df: pd.DataFrame, nome_arquivo: str = 'relatorio_transacoes.png'):
    """
    6. Visualização Gráfica Orientada a Objetos (Matplotlib):
    - Plota o valor total diário e a média móvel de 7 dias.
    - Garante limite do eixo Y iniciando em 0.
    """
    # Agrupamento diário
    df_diario = df.groupby(df['data_transacao'].dt.date)['valor'].sum().reset_index()
    df_diario['data_transacao'] = pd.to_datetime(df_diario['data_transacao'])
    df_diario['media_movel_7d'] = df_diario['valor'].rolling(window=7, min_periods=1).mean()

    # API Orientada a Objetos
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df_diario['data_transacao'], 
        df_diario['valor'], 
        label='Total Diário (R$)', 
        color='#1f77b4', 
        linewidth=1.5,
        alpha=0.7
    )
    ax.plot(
        df_diario['data_transacao'], 
        df_diario['media_movel_7d'], 
        label='Média Móvel (7 Dias)', 
        color='#ff7f0e', 
        linewidth=2.5
    )

    # Ajustes do gráfico
    ax.set_ylim(bottom=0)
    ax.set_title('Desempenho Diário de Transações e Média Móvel (7 Dias)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Data', fontsize=11)
    ax.set_ylabel('Valor Total (R$)', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()
    print(f"Gráfico salvo com sucesso como '{nome_arquivo}'.")


def main():
    print("=== Iniciando Pipeline de Dados Fintech ===")
    
    # Certifica que os dados foram criados
    if not os.path.exists('transacoes.csv'):
        print("Arquivo transacoes.csv não encontrado. Execute 'criar_dados.py' primeiro.")
        return

    # Passo 1
    df = carregar_e_limpar_transacoes('transacoes.csv')
    print(f"[1/6] Ingestão concluída. Registros: {len(df)}")

    # Passo 2
    df = pipeline_engenharia_dados(df)
    print(f"[2/6] Tratamento temporal e deduplicação concluídos. Registros restantes: {len(df)}")

    # Passo 3
    df_setembro_filtrado = filtrar_transacoes_setembro(df)
    print(f"[3/6] Transações filtradas em SP/RJ > R$ 5.000 em Setembro: {len(df_setembro_filtrado)}")

    # Passo 4
    df, pivot_risco = cruzar_dados_e_pivocar(df)
    print("\n--- Tabela Dinâmica: Total por Mês e Nível de Risco ---")
    print(pivot_risco)
    print("----------------------------------------------------\n")

    # Passo 5
    anomalias = detectar_outliers_zscore(df, threshold=2.5)
    print(f"[5/6] Detecção de anomalias concluída. Potenciais anomalias (Z > 2.5): {len(anomalias)}")

    # Passo 6
    gerar_relatorio_visual(df)
    print("=== Pipeline executado com sucesso ===")


if __name__ == '__main__':
    main()