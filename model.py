import torch
import torch.nn as nn
import torch.nn.functional as F




class STProtoPNet(nn.Module):
    def __init__(self, img_size, prototype_shape, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 threshold = 0.0029,
                 attention_num = 5):

        super(STProtoPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.threshold = threshold
        self.prototype_activation_function = prototype_activation_function  # log
        self.features = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.features.eval()
        self.attention_num = attention_num

        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        
        self.add_on_layers_trivial = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(in_channels=6, out_channels=self.prototype_shape[1], kernel_size =1),
            nn.Conv2d(in_channels=prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size =1)
        )

        self.add_on_layers_support = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(in_channels=6, out_channels=self.prototype_shape[1], kernel_size =1),
            nn.Conv2d(in_channels=prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size =1)
        )

        self.prototype_vectors_trivial = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.prototype_vectors_support = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.last_layer_trivial = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        self.last_layer_support = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        conv_features_trivial, conv_features_support = self.conv_features(x)
        cosine_similarities_trivial, cosine_similarities_support = self.prototype_distances(conv_features_trivial, conv_features_support)

        prototype_activations_trivial = self.global_max_pooling(cosine_similarities_trivial)
        prototype_activations_support = self.global_max_pooling(cosine_similarities_support)

        logits_trivial = self.last_layer_trivial(prototype_activations_trivial)
        logits_support = self.last_layer_support(prototype_activations_support)

        return (logits_trivial, logits_support), (prototype_activations_trivial, prototype_activations_support), \
               (cosine_similarities_trivial, cosine_similarities_support)

    def conv_features(self, x):
        x = self.get_attention(x)
        x_trivial = self.add_on_layers_trivial(x)     
        x_support = self.add_on_layers_support(x)
        return x_trivial, x_support

    def norm_img(self, imgs):
        max_val = imgs.max()
        min_val = imgs.min()
        imgs = (imgs - min_val) / (max_val - min_val)
        return imgs
    

    def get_attention(self, x):
        w_featmap = x.shape[-2] // 8
        h_featmap = x.shape[-1] // 8
        with torch.no_grad():
            attentions = self.features.get_last_selfattention(x)
            nh = attentions.shape[1]
            attentions = attentions[:, :, 0, 1:].reshape(x.shape[0], nh, -1)
            attentions = attentions.reshape(x.shape[0], nh, w_featmap, h_featmap)
            attentions = self.norm_img(attentions)
        return attentions

    def prototype_distances(self, conv_features_trivial, conv_features_support):
        cosine_similarities_trivial = self._cosine_convolution(self.prototype_vectors_trivial, conv_features_trivial)
        cosine_similarities_support = self._cosine_convolution(self.prototype_vectors_support, conv_features_support)
        # Relu from Deformable ProtoPNet: https://github.com/jdonnelly36/Deformable-ProtoPNet/blob/main/model.py
        ################################################
        cosine_similarities_trivial = torch.relu(cosine_similarities_trivial)
        cosine_similarities_support = torch.relu(cosine_similarities_support)
        ################################################

        return cosine_similarities_trivial, cosine_similarities_support

    def _cosine_convolution(self, prototypes, x):
        x = F.normalize(x, p=2, dim=1)
        prototype_vectors = F.normalize(prototypes, p=2, dim=1)
        similarity = F.conv2d(input=x, weight=prototype_vectors)
        return similarity


    def global_max_pooling(self, input):
        max_output = F.max_pool2d(input, kernel_size=(input.size()[2], input.size()[3]))
        max_output = max_output.view(-1, self.num_prototypes)
        return max_output

    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def distance_2_similarity_linear(self, distances):
        return (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]) ** 2 - distances

    def global_min_pooling(self, input):
        min_output = -F.max_pool2d(-input, kernel_size=(input.size()[2], input.size()[3]))
        min_output = min_output.view(-1, self.num_prototypes)
        return min_output
    
    def binarize_attention(self, attentions, threshold):
        binarized_attentions = torch.where(attentions > threshold, torch.tensor(1.0).to(attentions.device), torch.zeros_like(attentions))
        return binarized_attentions
    
    def push_forward_trivial(self, x):
        attentions = self.get_attention(x)
        conv_features_trivial, _ = self.conv_features(x)  # [batchsize, 64, 14, 14]

        similarities = self._cosine_convolution(self.prototype_vectors_trivial, conv_features_trivial)
        distances = - similarities

        conv_output = F.normalize(conv_features_trivial, p=2, dim=1)

        return conv_output, distances

    def push_forward_support(self, x):
        attentions = self.get_attention(x)
        _, conv_features_support = self.conv_features(x)  # [batchsize, 64, 14, 14]

        similarities = self._cosine_convolution(self.prototype_vectors_support, conv_features_support)
        distances = - similarities

        conv_output = F.normalize(conv_features_support, p=2, dim=1)

        return conv_output, distances


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer_trivial.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        self.last_layer_support.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers_trivial.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.add_on_layers_support.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)   # 0.0


def construct_STProtoPNet(pretrained=True, img_size=224,
                          prototype_shape=(2000, 6, 28, 28), num_classes=200,
                          prototype_activation_function='log'):

    return STProtoPNet(img_size=img_size,
                       prototype_shape=prototype_shape,
                       num_classes=num_classes,
                       init_weights=True,
                       prototype_activation_function=prototype_activation_function)
