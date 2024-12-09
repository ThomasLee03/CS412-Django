from django import forms
from .models import Image, Paper, ImageGenerated, Mask, CorruptedImage

class UploadImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image_file']
        widgets = {
            'image_file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

class CreatePaperForm(forms.ModelForm):
    class Meta:
        model = Paper
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control'}),
        }

class ImputationMethodForm(forms.Form):
    corrupted_image_id = forms.IntegerField(widget=forms.HiddenInput)
    imputation_method = forms.ChoiceField(
        choices=[
            ("mean", "Mean"),
            ("median", "Median"),
            ("mode", "Mode"),
            ("pca", "PCA"),
        ],
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    pca_preprocessing = forms.BooleanField(
        required=False,  # This makes the checkbox optional
        label="Apply PCA Preprocessing",
        initial=False  # Default value is False (unchecked)
    )
    class Meta:
        model = ImageGenerated
        fields = ["corrupted_image_id", "imputation_method"]


class UploadImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image_file']  # Use the actual field name defined in the model
    

class ImputeImageForm(forms.Form):
    # Filter the queryset to only show images uploaded by the currently logged-in user
    original_image = forms.ModelChoiceField(queryset=Image.objects.none())  # Default empty queryset

    imputation_method = forms.ChoiceField(choices=[
            ("mean", "Mean"),
            ("median", "Median"),
            ("mode", "Mode"),
            ("pca", "PCA"),
            ("total_variation", "Total Variation Inpainting"),  # Added this option
        ])
    
    pca_preprocessing = forms.BooleanField(
        required=False,  # This makes the checkbox optional
        label="Apply PCA Preprocessing",
        initial=False  # Default value is False (unchecked)
    )

    def __init__(self, *args, **kwargs):
        # Get the user from the request context
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

        # If the user is logged in, filter the queryset to only show their images
        if user is not None:
            self.fields['original_image'].queryset = Image.objects.filter(researcher=user.researcher_profile)