
# file: models.py
# author: Thomas Lee (tlee03@bu.edu), 11/22/2024


from django.db import models
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe
from django.utils import timezone
from django.urls import reverse

# Model for Researcher
class Researcher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="researcher_profile")
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    university = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.user.username})"
    
class Mask(models.Model):
    mask_file = models.ImageField(upload_to="masks/")
    researcher = models.ForeignKey(Researcher, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Mask {self.pk}"

class CorruptedImage(models.Model):
    corrupted_file = models.ImageField(upload_to="corrupted_images/")
    researcher = models.ForeignKey(Researcher, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Corrupted Image {self.pk}"

class Image(models.Model):
    image_file = models.ImageField(upload_to='images/')  # Make sure this field exists
    researcher = models.ForeignKey('Researcher', on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(default=timezone.now)
    mask = models.ForeignKey(Mask, null=True, blank=True, related_name="masked_images", on_delete=models.SET_NULL)
    corrupted_image = models.ForeignKey(CorruptedImage, null=True, blank=True, related_name="corrupted_images", on_delete=models.SET_NULL)

    def __str__(self):
        return f"Image {self.pk} - {self.researcher.user.username}"


# Model for ImageGenerated
class ImageGenerated(models.Model):
    image_file = models.ImageField(upload_to='images/generated/')
    imputation_method = models.CharField(max_length=200)
    ssim_value = models.FloatField(null=True, blank=True)  # SSIM value
    psnr_value = models.FloatField(null=True, blank=True)  # PSNR value
    researcher = models.ForeignKey(Researcher, on_delete=models.CASCADE, related_name="generated_images")
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    def __str__(self):
        return f"Generated Image {self.pk} - {self.imputation_method}"


# Model linking Original Image to Generated Image
class ImageWithGenerated(models.Model):
    original = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="generated_images")
    generated = models.ForeignKey(ImageGenerated, on_delete=models.CASCADE, related_name="original_image")

    def __str__(self):
        return f"Original {self.original.pk} -> Generated {self.generated.pk}"


# Intermediate model for Images in Paper
class PaperImage(models.Model):
    paper = models.ForeignKey("Paper", on_delete=models.CASCADE)
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    order = models.PositiveIntegerField()  # 1, 2, 3, etc.

    class Meta:
        unique_together = ('paper', 'order')

    def __str__(self):
        return f"Paper {self.paper.pk} - Image {self.image.pk} (Order {self.order})"
    

# Intermediate model for Generated Images in Paper
class PaperWithGeneratedImage(models.Model):
    paper = models.ForeignKey("Paper", on_delete=models.CASCADE)
    generated_image = models.ForeignKey(ImageGenerated, on_delete=models.CASCADE)  # Change 'image' to 'generated_image'
    order = models.PositiveIntegerField()  # GEN_1, GEN_2, etc.

    class Meta:
        unique_together = ('paper', 'order')

    def __str__(self):
        return f"Paper {self.paper.pk} - Generated Image {self.generated_image.pk} (Order {self.order})"

# Intermediate model for Corrupted Images in Paper

class PaperWithCorruptedImage(models.Model):
    paper = models.ForeignKey("Paper", on_delete=models.CASCADE)
    corrupted_image = models.ForeignKey(CorruptedImage, on_delete=models.CASCADE)  # Change 'image' to 'corrupted_image'
    order = models.PositiveIntegerField()  # CORR_1, CORR_2, etc.

    class Meta:
        unique_together = ('paper', 'order')

    def __str__(self):
        return f"Paper {self.paper.pk} - Corrupted Image {self.corrupted_image.pk} (Order {self.order})"


# Intermediate model for Masks in Paper
class PaperWithMask(models.Model):
    paper = models.ForeignKey("Paper", on_delete=models.CASCADE)
    mask = models.ForeignKey(Mask, on_delete=models.CASCADE)
    order = models.PositiveIntegerField()  # MASK_1, MASK_2, etc.

    class Meta:
        unique_together = ('paper', 'order')

    def __str__(self):
        return f"Paper {self.paper.pk} - Mask {self.mask.pk} (Order {self.order})"

# Model for Paper
class Paper(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField(help_text="Use [% N %] to embed images, [% GEN_N %] for generated images, [% MASK_N %] for masks, or [% CORR_N %] for corrupted images.")
    time_created = models.DateTimeField(auto_now_add=True)
    images = models.ManyToManyField(Image, through="PaperImage", related_name="papers", blank=True)
    generated_images = models.ManyToManyField(ImageGenerated, through="PaperWithGeneratedImage", related_name="papers", blank=True)
    corrupted_images = models.ManyToManyField(CorruptedImage, through="PaperWithCorruptedImage", related_name="papers", blank=True)
    masks = models.ManyToManyField(Mask, through="PaperWithMask", related_name="papers", blank=True)

    def __str__(self):
        return f"{self.title}"

    class Meta:
        ordering = ['-time_created']

    def get_absolute_url(self):
        return reverse('paper_detail', kwargs={'pk': self.pk})

    def render_content(self):
        """
        Replace placeholders in the content with corresponding image tags.
        Each type of image is displayed on its own line.
        """
        rendered_content = self.content

        # Replace standard images
        for paper_image in self.paperimage_set.all():
            placeholder = f"[% {paper_image.order} %]"
            if placeholder in rendered_content:
                image_tag = f'<br><img src="{paper_image.image.image_file.url}" alt="Image {paper_image.order}"><br>'
                rendered_content = rendered_content.replace(placeholder, image_tag)

        # Replace generated images
        for paper_gen_image in self.paperwithgeneratedimage_set.all():
            placeholder = f"[% GEN_{paper_gen_image.order} %]"
            if placeholder in rendered_content:
                image_tag = f'<br><img src="{paper_gen_image.generated_image.image_file.url}" alt="Generated Image {paper_gen_image.order}"><br>'
                rendered_content = rendered_content.replace(placeholder, image_tag)

        # Replace corrupted images
        for paper_corr_image in self.paperwithcorruptedimage_set.all():
            placeholder = f"[% CORR_{paper_corr_image.order} %]"
            if placeholder in rendered_content:
                image_tag = f'<br><img src="{paper_corr_image.corrupted_image.corrupted_file.url}" alt="Corrupted Image {paper_corr_image.order}"><br>'
                rendered_content = rendered_content.replace(placeholder, image_tag)

        # Replace masks
        for paper_mask in self.paperwithmask_set.all():
            placeholder = f"[% MASK_{paper_mask.order} %]"
            if placeholder in rendered_content:
                image_tag = f'<br><img src="{paper_mask.mask.mask_file.url}" alt="Mask {paper_mask.order}"><br>'
                rendered_content = rendered_content.replace(placeholder, image_tag)

        return mark_safe(rendered_content)






# Model linking Paper to Researcher (multiple researchers can be associated with a single paper)
class PaperWithResearcher(models.Model):
    researcher = models.ForeignKey(Researcher, on_delete=models.CASCADE)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)

    def __str__(self):
        return f"Paper '{self.paper.title}' by {self.researcher}"

