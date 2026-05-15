import torch

def get_batch(
        data: torch.Tensor,
        batch_size: int, 
        block_size: int,
        device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """다음 토큰 예측 학습용 mini-batch를 생성한다.

    Args:
        data: 전체 토큰 ID. shape: (num_token,)
        batch_size: 한 번에 뽑을 샘플 개수.
        block_size: 각 샘플의 문맥 길이.
        device: batch를 올릴 장치. 예: "cpu", "mps", "cuda"
    
    Returns:
        x: 입력 토큰 batch. shape: (batch_size, block_size)
        y: 정답 토큰 batch. shape: (batch_size, block_size)
    예:
        data = [0, 1, 2, 3, 4]
        block_size = 3

        x = [0, 1, 2]
        y = [1, 2, 3]
    """
    
    if data.ndim != 1:
        raise ValueError(f"data must be 1D tensor, got shape {tuple(data.shape)}")
    
    if len(data) <= block_size:
        raise ValueError(f"data length must be greater than block_size: "
                        f"len(data)={len(data)}, block_size={block_size}")
    
    max_start = len(data) - block_size - 1
    
    # 전체 데이터에서 랜덤 시작 위치를 batch_size개 뽑고
    # 그 다음 각 시작 위치마다 길이 block_size만큼 잘라서 x를 만들고
    # 한 칸 뒤를 잘라서 y를 만듭니다.
    starts = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

    return x.to(device), y.to(device)