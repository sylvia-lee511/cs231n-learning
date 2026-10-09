import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


torch.manual_seed(0)


# ==========================================
# 1. Small network
# ==========================================

class SmallEncoder(nn.Module):

    def __init__(
        self,
        input_dim=16,
        hidden_dim=32,
        output_dim=8
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

student = SmallEncoder()

teacher = copy.deepcopy(
    student
)


for param in teacher.parameters():

    # TODO 1
    # freeze teacher
    param.requires_grad = False

output_dim = 8

center = torch.zeros(
    1,
    output_dim
)

center_momentum = 0.9




B = 8
D = 16

x = torch.randn(B, D)

view1 = (
    x
    +
    0.1 * torch.randn_like(x)
)

view2 = (
    x
    +
    0.1 * torch.randn_like(x)
)

student_logits = student(
    view1
)

print("old_center:", center)


with torch.no_grad():

    teacher_logits = teacher(
        view2
    )

student_temp = 0.1
teacher_temp = 0.04

student_log_prob = F.log_softmax(
    student_logits / student_temp,
    dim=1
)

teacher_prob = F.softmax(
    (teacher_logits-center) / teacher_temp,
    dim=1
)

print("Student log prob:")
print(student_log_prob)

print("Teacher prob:")
print(teacher_prob)

with torch.no_grad():

    batch_center = teacher_logits.mean(
        dim=0,
        keepdim=True
    )

    center = (
        center_momentum * center
        +
        (1 - center_momentum)
        * batch_center
    )

print("new_center:", center)

old_teacher = [
    p.clone()
    for p in teacher.parameters()
]

# TODO 2
#
# 对每个样本:
#
# - sum(
#     teacher_prob
#     *
#     student_log_prob
#   )
#
# 然后 batch mean


loss = (
    - (teacher_prob * student_log_prob)
).sum(dim=1).mean()

optimizer = torch.optim.SGD(
    student.parameters(),
    lr=0.01
)


optimizer.zero_grad()

loss.backward()

optimizer.step()

momentum = 0.99


with torch.no_grad():

    for student_param, teacher_param in zip(
        student.parameters(),
        teacher.parameters()
    ):

        # TODO 3
        #
        # teacher =
        # m * teacher
        # +
        # (1-m) * student

        teacher_param.data = (
            momentum * teacher_param.data
            +
            (1 - momentum) * student_param.data
        )

for old, new in zip(
    old_teacher,
    teacher.parameters()
):

    print(
        (new - old).norm()
    )